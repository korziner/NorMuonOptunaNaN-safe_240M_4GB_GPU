#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
if "--help" in sys.argv or "-h" in sys.argv:
    print("Использование: python NorMuon_hybrid.py [опции]")
    print("  --use_normuon         Использовать гибридный оптимизатор (NorMuon для матриц, AdamW для векторов)")
    print("  --disable_muon        Использовать AdamW для всех параметров")
    print("  --optimizer_8bit      Использовать 8-битный AdamW (bitsandbytes) для экономии VRAM")
    print("  --learning_rate LR    Скорость обучения")
    print("  --lr_matrix LR        LR для NorMuon (если --use_normuon)")
    print("  --lr_vector LR        LR для AdamW векторов (если --use_normuon)")
    print("  --output_dir DIR      ...")
    print("  --use_optuna          Использовать Optuna для поиска гиперпараметров")
    print("  --optuna_trials N     Количество испытаний")
    print("  --optuna_max_steps N  Шагов на trial (по умолч. 15)")
    print("  --optuna_dataset_samples N  Размер выборки для Optuna (по умолч. 500)")
    print("  --optuna_nan_retries N     Число повторений при NaN (по умолч. 3)")
    print("  --optuna_nan_reduction_factor F  Множитель уменьшения LR (по умолч. 0.5)")
    print("  --optuna_tmp_dir DIR  Временная папка (по умолч. /dev/shm/optuna_tmp)")
    print("  --attention_backend {eager,sdpa}  Бэкенд внимания (по умолч. eager)")
    sys.exit(0)

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import re
import time
import torch
import gc
import shutil
import math
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import SFTTrainer, SFTConfig

# ---------- 1. Muon (оставляем как есть) ----------
def zeropower_via_newtonschulz5(G, steps=5):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            closure()
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)
                if p.ndim >= 2:
                    g = zeropower_via_newtonschulz5(g)
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                p.data.add_(buf, alpha=-lr)

# ---------- 2. Гибридный оптимизатор (NorMuon + AdamW) ----------
try:
    from dion import NorMuon
    DION_AVAILABLE = True
except ImportError:
    DION_AVAILABLE = False
    print("⚠️ dion не установлен. Установите: pip install git+https://github.com/microsoft/dion.git")

class HybridOptimizer(torch.optim.Optimizer):
    def __init__(self, matrix_params, vector_params, lr_matrix=5e-7, lr_vector=2e-7, betas=(0.9, 0.95), weight_decay=0.01):
        self.matrix_optim = NorMuon(matrix_params, lr=lr_matrix, betas=betas, weight_decay=weight_decay)
        self.vector_optim = torch.optim.AdamW(vector_params, lr=lr_vector, weight_decay=weight_decay)
        self.param_groups = self.matrix_optim.param_groups + self.vector_optim.param_groups
        self.defaults = {}

    def step(self, closure=None):
        self.matrix_optim.step(closure)
        self.vector_optim.step(closure)

    def state_dict(self):
        return {'matrix': self.matrix_optim.state_dict(), 'vector': self.vector_optim.state_dict()}

    def load_state_dict(self, state_dict):
        self.matrix_optim.load_state_dict(state_dict['matrix'])
        self.vector_optim.load_state_dict(state_dict['vector'])

    def zero_grad(self, set_to_none=True):
        self.matrix_optim.zero_grad(set_to_none)
        self.vector_optim.zero_grad(set_to_none)

    def add_param_group(self, group):
        raise NotImplementedError("HybridOptimizer does not support adding param groups after init")

# ---------- 3. Данные ----------
def load_data(data_path):
    def remove_system_prompt(text):
        return re.sub(r'<\|system\|>.*?<\|user\|>', '<|user|>', text, flags=re.DOTALL)
    data_list = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                if 'text' in obj and obj['text'].strip():
                    cleaned = remove_system_prompt(obj['text']).strip()
                    if cleaned:
                        data_list.append({"text": cleaned})
            except:
                continue
    return Dataset.from_dict({"text": [item['text'] for item in data_list]})

# ---------- 4. Callback для логов ----------
class PrettyLoggingCallback(TrainerCallback):
    def __init__(self, output_dir, total_steps, log_every=10):
        self.output_dir = output_dir
        self.total_steps = total_steps
        self.log_every = log_every
        self.log_path = os.path.join(output_dir, "training_log.jsonl")
        self.pretrain_log_path = os.path.join(output_dir, "pretrain.log")
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                pass
        if not os.path.exists(self.pretrain_log_path):
            with open(self.pretrain_log_path, 'w') as f:
                pass
        self.start_time = time.time()
        self.batch_size = None
        self.grad_accum = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        step = state.global_step
        if step % self.log_every != 0:
            return
        if not logs:
            return

        loss = logs.get('loss')
        grad_norm = logs.get('grad_norm')
        lr = logs.get('learning_rate')
        ema_loss = logs.get('ema_loss', loss)
        accuracy = logs.get('mean_token_accuracy')
        entropy = logs.get('entropy')

        if self.batch_size is None:
            self.batch_size = args.per_device_train_batch_size
            self.grad_accum = args.gradient_accumulation_steps

        if hasattr(state, 'num_input_tokens_seen') and state.num_input_tokens_seen > 0:
            num_tokens = state.num_input_tokens_seen
        else:
            max_length = getattr(args, 'max_length', 512)
            num_tokens = step * self.batch_size * self.grad_accum * max_length

        elapsed = time.time() - self.start_time
        tok_per_sec = num_tokens / elapsed if elapsed > 0 else None

        vram_free_gb = None
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            vram_free_gb = free / (1024 ** 3)

        parts = []
        parts.append(f"step {step}/{self.total_steps}")
        if loss is not None:
            parts.append(f"loss {loss:.4f}")
        if ema_loss is not None:
            parts.append(f"ema {ema_loss:.4f}")
        if lr is not None:
            parts.append(f"lr {lr:.2e}")
        if grad_norm is not None:
            parts.append(f"gnorm {grad_norm:.2f}")
        if tok_per_sec is not None:
            parts.append(f"{tok_per_sec:>8.0f} tok/s")
        if vram_free_gb is not None:
            parts.append(f"VRAM {vram_free_gb:.2f} GB free")
        if accuracy is not None:
            parts.append(f"acc {accuracy:.3f}")
        if entropy is not None:
            parts.append(f"entropy {entropy:.3f}")

        log_line = " │ ".join(parts)
        print(log_line)
        with open(self.pretrain_log_path, 'a') as f:
            f.write(log_line + '\n')
            f.flush()

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "loss": float(loss) if loss is not None else None,
            "ema_loss": float(ema_loss) if ema_loss is not None else None,
            "grad_norm": float(grad_norm) if grad_norm is not None else None,
            "learning_rate": float(lr) if lr is not None else None,
            "mean_token_accuracy": float(accuracy) if accuracy is not None else None,
            "entropy": float(entropy) if entropy is not None else None,
            "tokens_per_second": tok_per_sec,
            "vram_free_gb": vram_free_gb,
            "num_tokens_seen": num_tokens,
            "epoch": float(state.epoch)
        }
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            f.flush()

# ---------- 5. Callback для лучшей модели (с защитой от NaN/0 loss) ----------
class BestModelCallback(TrainerCallback):
    def __init__(self, output_dir, tokenizer):
        self.output_dir = output_dir
        self.best_model_path = os.path.join(output_dir, "best_model")
        self.tokenizer = tokenizer
        self.best_loss_file = os.path.join(self.best_model_path, "best_loss.json")
        self.best_loss = self._load_best_loss()
        os.makedirs(self.best_model_path, exist_ok=True)

    def _load_best_loss(self):
        if os.path.exists(self.best_loss_file):
            try:
                with open(self.best_loss_file, 'r') as f:
                    data = json.load(f)
                    return data.get('best_loss', float('inf'))
            except:
                return float('inf')
        return float('inf')

    def _save_best_loss(self):
        with open(self.best_loss_file, 'w') as f:
            json.dump({'best_loss': self.best_loss}, f)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        loss = logs.get('loss')
        if loss is None or not isinstance(loss, (float, int)):
            return
        # Защита от плохих значений
        if math.isnan(loss) or loss == 0.0:
            print(f"⚠️ Not updating best model: invalid loss {loss}")
            return
        if loss < self.best_loss:
            self.best_loss = loss
            self._save_best_loss()
            model = kwargs.get('model')
            if model is not None:
                print(f"\n🏆 New best loss: {loss:.4f} — saving to {self.best_model_path}")
                model.save_pretrained(self.best_model_path)
                self.tokenizer.save_pretrained(self.best_model_path)
                torch.cuda.empty_cache()

# ---------- 6. Callback для чекпоинтов (с защитой от плохих loss) ----------
class WeightsOnlyCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir, save_steps, tokenizer, min_free_gb=2):
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.tokenizer = tokenizer
        self.min_free_gb = min_free_gb
        os.makedirs(output_dir, exist_ok=True)

    def _free_disk_space(self, required_gb):
        free = shutil.disk_usage(self.output_dir).free // (2**30)
        if free >= required_gb:
            return True
        ckpts = [d for d in os.listdir(self.output_dir) if d.startswith("checkpoint-")]
        ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(self.output_dir, x)))
        for ckpt in ckpts:
            if free >= required_gb:
                break
            path = os.path.join(self.output_dir, ckpt)
            shutil.rmtree(path, ignore_errors=True)
            free = shutil.disk_usage(self.output_dir).free // (2**30)
        return free >= required_gb

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step % self.save_steps == 0 and step > 0:
            # Проверяем последний loss
            last_log = state.log_history[-1] if state.log_history else {}
            loss = last_log.get('loss')
            if loss is not None and (math.isnan(loss) or loss == 0.0):
                print(f"⚠️ Skipping checkpoint at step {step} due to invalid loss: {loss}")
                return control
            torch.cuda.empty_cache()
            gc.collect()
            if not self._free_disk_space(self.min_free_gb):
                print(f"❌ Disk full, skipping checkpoint {step}")
                return control
            ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            try:
                model.save_pretrained(ckpt_dir)
                self.tokenizer.save_pretrained(ckpt_dir)
                if state.log_history:
                    with open(os.path.join(ckpt_dir, "trainer_state.json"), "w") as f:
                        json.dump({"global_step": step, "log_history": state.log_history[-100:]}, f)
                print(f"\n💾 Checkpoint saved: {ckpt_dir}")
            except Exception as e:
                print(f"⚠️ Failed to save checkpoint: {e}")
            torch.cuda.empty_cache()
            gc.collect()
        return control

class InferenceCallback(TrainerCallback):
    def __init__(self, tokenizer, prompts, every=500):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.every = every

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step % self.every == 0 and step > 0:
            print(f"\n🧪 [Step {step}] Test inference:")
            model.eval()
            for prompt in self.prompts:
                full_prompt = f"<|user|>\n{prompt}\n<|assistant|>"
                inputs = self.tokenizer(full_prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.7)
                response = self.tokenizer.decode(out[0], skip_special_tokens=True)
                if full_prompt in response:
                    response = response.split(full_prompt)[-1].strip()
                print(f"Q: {prompt}\nA: {response}\n")
            model.train()
            torch.cuda.empty_cache()

# ---------- 7. Self-healing callback для NaN (уменьшает LR) ----------
class NaNSelfHealingCallback(TrainerCallback):
    def __init__(self, optimizer, lr_reduction_factor=0.5, max_retries=3):
        self.optimizer = optimizer
        self.lr_reduction_factor = lr_reduction_factor
        self.max_retries = max_retries
        self.nan_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        loss = logs.get('loss')
        grad_norm = logs.get('grad_norm')
        has_nan = (loss is not None and (math.isnan(loss) or math.isinf(loss))) or \
                  (grad_norm is not None and (math.isnan(grad_norm) or math.isinf(grad_norm)))
        if has_nan:
            if self.nan_count >= self.max_retries:
                raise RuntimeError(f"NaN persists after {self.max_retries} LR reductions, aborting.")
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = old_lr * self.lr_reduction_factor
                param_group['lr'] = new_lr
                print(f"⚠️ NaN detected (retry {self.nan_count+1}/{self.max_retries})! Reducing LR from {old_lr:.2e} to {new_lr:.2e}")
            self.nan_count += 1
            torch.cuda.empty_cache()

# ---------- 8. Парсинг аргументов ----------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=r"/content/training_data_19century.json")
    parser.add_argument("--model_name", type=str, default="oopere/gemma-3-270m-14L-distilled")
    parser.add_argument("--output_dir", type=str, default="./model_checkpoints")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-7)
    parser.add_argument("--lr_matrix", type=float, default=5e-7, help="LR for NorMuon (if --use_normuon)")
    parser.add_argument("--lr_vector", type=float, default=2e-7, help="LR for AdamW on vectors (if --use_normuon)")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")
    parser.add_argument("--skip_benchmarks", action="store_true")
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--disable_qat", action="store_true")
    parser.add_argument("--disable_muon", action="store_true")
    parser.add_argument("--use_normuon", action="store_true", help="Use HybridOptimizer (NorMuon matrices + AdamW vectors)")
    parser.add_argument("--optimizer_8bit", action="store_true", help="Use 8-bit AdamW (bitsandbytes) to save VRAM")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--attention_backend", type=str, default="eager", choices=["eager", "sdpa"], help="Attention backend")
    parser.add_argument("--use_optuna", action="store_true", help="Use Optuna for hyperparameter search")
    parser.add_argument("--optuna_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--optuna_study_dir", type=str, default="./optuna_study", help="Directory for Optuna study")
    parser.add_argument("--optuna_max_steps", type=int, default=15, help="Max steps per Optuna trial")
    parser.add_argument("--optuna_logging_steps", type=int, default=5, help="Logging steps per Optuna trial")
    parser.add_argument("--optuna_dataset_samples", type=int, default=500, help="Number of samples for Optuna tuning")
    parser.add_argument("--optuna_tmp_dir", type=str, default="/dev/shm/optuna_tmp", help="Temp dir for Optuna trials")
    parser.add_argument("--optuna_nan_retries", type=int, default=3, help="How many times to retry a trial if NaN occurs")
    parser.add_argument("--optuna_nan_reduction_factor", type=float, default=0.5, help="Multiply learning rate by this factor on each NaN retry")
    parser.add_argument("--optuna_nan_reduce_gradnorm", action="store_true", default=True, help="Also reduce max_grad_norm on NaN")
    return parser.parse_args()

# ---------- 9. Optuna objective (с защитой от нулевого loss и NaN) ----------
def optuna_objective(trial, model, tokenizer, tokenized_dataset, base_args):
    import optuna
    from optuna.exceptions import TrialPruned

    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4])
    grad_accum = trial.suggest_categorical("grad_accum", [1, 2])
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 5e-6, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.01, 0.2)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 2.0)
    lr_scheduler = trial.suggest_categorical("lr_scheduler", ["cosine", "linear"])

    print(f"\n🔬 Trial {trial.number}: bs={batch_size}, ga={grad_accum}, lr={learning_rate:.2e}, warmup={warmup_ratio:.3f}, grad_clip={max_grad_norm:.2f}, sched={lr_scheduler}")

    trial_tmp_dir = os.path.join(base_args.optuna_tmp_dir, f"trial_{trial.number}")
    os.makedirs(trial_tmp_dir, exist_ok=True)

    def run_with_params(bs, ga, lr, grad_clip):
        # Создаём оптимизатор
        if base_args.optimizer_8bit:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.01)
            except Exception as e:
                print(f"⚠️ bitsandbytes failed: {e}, falling back to AdamW")
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        test_config = SFTConfig(
            output_dir=trial_tmp_dir,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=ga,
            max_length=base_args.max_length,
            packing=False,
            learning_rate=lr,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="constant",
            max_grad_norm=grad_clip,
            fp16=base_args.fp16,
            bf16=False,
            max_steps=base_args.optuna_max_steps,
            logging_steps=base_args.optuna_logging_steps,
            dataset_text_field=None,
            report_to="none",
            remove_unused_columns=False,
        )
        trainer = SFTTrainer(
            model=model,
            args=test_config,
            train_dataset=tokenized_dataset,
            processing_class=tokenizer,
            optimizers=(optimizer, None),
        )
        trainer.train()
        logs = trainer.state.log_history
        del trainer
        torch.cuda.empty_cache()
        gc.collect()

        # Проверка на NaN
        has_nan = False
        for log in logs:
            loss_val = log.get('loss')
            grad_val = log.get('grad_norm')
            if (loss_val is not None and math.isnan(loss_val)) or (grad_val is not None and math.isnan(grad_val)):
                has_nan = True
                break
        if has_nan:
            raise ValueError("NaN_detected")

        final_loss = None
        for log in reversed(logs):
            if "loss" in log:
                final_loss = log["loss"]
                break
        if final_loss is None or final_loss > 1e5 or final_loss == 0.0:
            raise ValueError("Invalid loss (zero or too large)")

        if len(logs) > 1:
            steps = logs[-1].get("step", base_args.optuna_max_steps)
            elapsed = logs[-1].get("timestamp", 0) - logs[0].get("timestamp", 0) if "timestamp" in logs[0] else 10
            tok_per_sec = steps * bs * ga * base_args.max_length / max(elapsed, 0.1)
        else:
            tok_per_sec = 0
        score = final_loss - 0.0001 * tok_per_sec
        if any(log.get("grad_norm", 0) > 100 for log in logs):
            score += 0.5
        return score

    current_lr = learning_rate
    current_grad_clip = max_grad_norm
    for retry in range(base_args.optuna_nan_retries + 1):
        try:
            result = run_with_params(batch_size, grad_accum, current_lr, current_grad_clip)
            return result
        except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as e:
            err_str = str(e)
            if "NaN" in err_str or "nan" in err_str or "OOM" in err_str or "Invalid loss" in err_str:
                if retry >= base_args.optuna_nan_retries:
                    print(f"   ❌ Trial {trial.number}: {err_str} after {retry} retries. Pruning.")
                    trial.set_user_attr("pruned", f"{err_str} after retries")
                    raise TrialPruned(f"Persistent {err_str}")
                new_lr = current_lr * base_args.optuna_nan_reduction_factor
                print(f"   ⚠️ {err_str} detected (retry {retry+1}/{base_args.optuna_nan_retries}). Reducing LR from {current_lr:.2e} to {new_lr:.2e}")
                current_lr = new_lr
                if base_args.optuna_nan_reduce_gradnorm:
                    new_grad_clip = current_grad_clip * base_args.optuna_nan_reduction_factor
                    print(f"      Also reducing max_grad_norm from {current_grad_clip:.2f} to {new_grad_clip:.2f}")
                    current_grad_clip = new_grad_clip
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                if retry == 0:
                    new_lr = current_lr * base_args.optuna_nan_reduction_factor
                    print(f"   ⚠️ Unstable: {e}. Retrying with LR reduced to {new_lr:.2e}")
                    current_lr = new_lr
                    continue
                else:
                    raise
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}. Pruning.")
            raise TrialPruned(f"Unexpected error: {e}")
    # Fallback
    raise TrialPruned("Exhausted retries")

# ---------- 10. Main ----------
def main():
    args = parse_args()

    # Проверка learning rate
    if args.learning_rate == 0.0:
        print("⚠️ LR is zero, resetting to 1e-7")
        args.learning_rate = 1e-7

    print(f"📥 Loading data from {args.data_path}...")
    dataset = load_data(args.data_path)
    print(f"✅ Loaded {len(dataset)} examples")

    print(f"📦 Loading model {args.model_name}...")
    attn_impl = args.attention_backend if args.attention_backend != "eager" else None
    if args.resume_from and not args.inference_only:
        tokenizer = AutoTokenizer.from_pretrained(args.resume_from, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.resume_from, trust_remote_code=True,
            attn_implementation=attn_impl
        ).to("cuda")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, trust_remote_code=True,
            attn_implementation=attn_impl
        ).to("cuda")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Проверка на NaN веса при возобновлении
    if args.resume_from:
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"❌ Model parameter {name} contains NaN! Aborting. Remove corrupted checkpoint.")
                sys.exit(1)

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print(f"✅ Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    if args.skip_benchmarks:
        print("⏩ Benchmarks disabled (--skip_benchmarks)")
    else:
        print("\n⏩ Skipping benchmarks")

    # QAT
    qat_config = None
    if not args.disable_qat and not args.inference_only:
        try:
            from torchao.quantization import quantize_
            from torchao.quantization.qat import QATConfig
            from torchao.quantization import Int8DynamicActivationInt4WeightConfig
            qat_config = Int8DynamicActivationInt4WeightConfig(group_size=32)
            quantize_(model, QATConfig(qat_config, step="prepare"))
            print("✅ QAT prepare applied")
        except ImportError:
            print("⚠️ torchao not installed, skipping QAT")

    if args.inference_only:
        if not args.resume_from:
            print("Error: --inference_only requires --resume_from")
            return
        prompt = args.prompt or "Какие три основных цвѣта?"
        print(f"\n🔮 Inference from {args.resume_from}")
        inputs = tokenizer(f"<|user|>\n{prompt}\n<|assistant|>", return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
        print(tokenizer.decode(out[0], skip_special_tokens=True))
        return

    # ---------- Подготовка для Optuna: токенизированный датасет один раз ----------
    if args.use_optuna:
        num_samples = min(args.optuna_dataset_samples, len(dataset))
        small_raw = dataset.select(range(num_samples))
        def tokenize_func(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=args.max_length,
                padding=False,
                return_tensors=None,
            )
        tokenized_small = small_raw.map(tokenize_func, batched=True, remove_columns=small_raw.column_names)
        tokenized_small.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        print(f"📦 Pre-tokenized dataset for Optuna: {len(tokenized_small)} examples")
    else:
        tokenized_small = None

    # ---------- Optuna hyperparameter search ----------
    if args.use_optuna:
        import optuna
        from optuna.exceptions import TrialPruned
        import shutil

        free_gb = shutil.disk_usage(args.output_dir).free // (2**30)
        if free_gb < 1:
            print(f"⚠️ Low disk space: {free_gb} GB. Cleaning old checkpoints (except checkpoint-30)...")
            for item in sorted(os.listdir(args.output_dir)):
                if item.startswith("checkpoint-") and item != "checkpoint-30":
                    full = os.path.join(args.output_dir, item)
                    if os.path.isdir(full):
                        shutil.rmtree(full, ignore_errors=True)
            free_gb = shutil.disk_usage(args.output_dir).free // (2**30)
        if free_gb < 0.5:
            print("❌ Still low disk space. Switching to in-memory Optuna storage.")
            use_disk_storage = False
        else:
            use_disk_storage = True

        os.makedirs(args.optuna_study_dir, exist_ok=True)
        db_path = os.path.join(args.optuna_study_dir, "study.db")

        if use_disk_storage:
            try:
                if os.path.exists(db_path) and os.path.getsize(db_path) > 200*1024*1024:
                    print(f"⚠️ Optuna DB too large, deleting")
                    os.remove(db_path)
                study = optuna.create_study(
                    direction="minimize",
                    study_name="noromuon_hybrid",
                    storage=f"sqlite:///{db_path}",
                    load_if_exists=True
                )
            except Exception as e:
                if "disk is full" in str(e) or "database or disk is full" in str(e):
                    print("⚠️ Disk full, falling back to in-memory study")
                    use_disk_storage = False
                else:
                    raise
        if not use_disk_storage:
            study = optuna.create_study(direction="minimize", study_name="noromuon_hybrid")

        from functools import partial
        objective_partial = partial(optuna_objective, model=model, tokenizer=tokenizer, tokenized_dataset=tokenized_small, base_args=args)
        study.optimize(objective_partial, n_trials=args.optuna_trials, timeout=3600, catch=(Exception,))
        best_params = study.best_params
        print("🏆 Best parameters found:", best_params)
        args.batch_size = best_params.get("batch_size", args.batch_size)
        args.grad_accum = best_params.get("grad_accum", args.grad_accum)
        args.learning_rate = best_params.get("learning_rate", args.learning_rate)
        args.warmup_ratio = best_params.get("warmup_ratio", args.warmup_ratio)
        args.max_grad_norm = best_params.get("max_grad_norm", args.max_grad_norm)
        best_lr_scheduler = best_params.get("lr_scheduler", "cosine")
        print(f"✅ Optuna search completed. Using batch_size={args.batch_size}, grad_accum={args.grad_accum}, lr={args.learning_rate:.2e}, warmup={args.warmup_ratio}, max_grad_norm={args.max_grad_norm}, scheduler={best_lr_scheduler}")
        # Ещё раз убедимся, что learning rate не нулевой
        if args.learning_rate == 0.0:
            args.learning_rate = 1e-7
            print("⚠️ LR was zero after Optuna, reset to 1e-7")
    else:
        best_lr_scheduler = "cosine"

    # ---------- Optimizer selection ----------
    if args.optimizer_8bit:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
            print(f"🔧 Using 8-bit AdamW optimizer (lr={args.learning_rate})")
        except Exception as e:
            print(f"⚠️ bitsandbytes failed: {e}. Falling back to AdamW.")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    elif args.use_normuon and DION_AVAILABLE:
        matrix_params = [p for p in model.parameters() if p.ndim >= 2]
        vector_params = [p for p in model.parameters() if p.ndim < 2]
        print(f"📊 Matrices: {len(matrix_params)} parameters, Vectors: {len(vector_params)} parameters")
        optimizer = HybridOptimizer(
            matrix_params, vector_params,
            lr_matrix=args.lr_matrix,
            lr_vector=args.lr_vector,
            betas=(0.9, 0.95),
            weight_decay=0.01
        )
        print(f"🔧 Using HybridOptimizer (NorMuon lr={args.lr_matrix}, AdamW lr={args.lr_vector})")
    elif not args.disable_muon:
        optimizer = Muon(model.parameters(), lr=args.learning_rate, momentum=0.95, weight_decay=0.01)
        print(f"🔧 Using Muon optimizer (lr={args.learning_rate})")
    else:
        optimizer = None
        print(f"🔧 Using AdamW (lr={args.learning_rate})")

    # ---------- SFTConfig ----------
    training_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        packing=False,
        learning_rate=args.learning_rate,
        lr_scheduler_type=best_lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        fp16=args.fp16,
        bf16=False,
        logging_steps=10,
        save_steps=1_000_000,
        save_total_limit=1,
        dataset_text_field="text",
        report_to="none",
        remove_unused_columns=False,
    )

    # ---------- Callbacks ----------
    total_steps_estimate = 1000000
    pretty_logger = PrettyLoggingCallback(args.output_dir, total_steps_estimate, log_every=10)
    best_model_cb = BestModelCallback(args.output_dir, tokenizer)
    checkpoint_cb = WeightsOnlyCheckpointCallback(args.output_dir, args.save_steps, tokenizer)
    inference_cb = InferenceCallback(tokenizer, ["Какие три основных цвѣта?", "Кто ты?"], every=100)
    nan_healing_cb = NaNSelfHealingCallback(optimizer, lr_reduction_factor=0.5, max_retries=3)

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        optimizers=(optimizer, None) if optimizer else (None, None),
        callbacks=[checkpoint_cb, inference_cb, pretty_logger, best_model_cb, nan_healing_cb]
    )

    if hasattr(trainer.state, 'max_steps') and trainer.state.max_steps > 0:
        pretty_logger.total_steps = trainer.state.max_steps

    print("\n🚀 Starting training...")
    trainer.train()

    print("\n💾 Saving final model...")
    if not args.disable_qat and qat_config is not None:
        try:
            from torchao.quantization import quantize_
            from torchao.quantization.qat import QATConfig
            quantize_(model, QATConfig(qat_config, step="convert"))
            print("✅ QAT convert applied")
        except:
            pass
    trainer.save_model(f"{args.output_dir}_final")
    tokenizer.save_pretrained(f"{args.output_dir}_final")
    print(f"✅ Training finished. Model saved to {args.output_dir}_final")

if __name__ == "__main__":
    main()
