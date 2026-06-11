# NorMuonOptunaNaN-safe_240M_4GB_GPU
Continued full pretrain for gemma3 270M on 4GB GPU

--attention_backend {eager,sdpa}  Бэкенд внимания (по умолч. eager)

<img width="1175" height="703" alt="image" src="https://github.com/user-attachments/assets/40270859-cbe0-4bdd-aa29-ec20d1e0d1a8" />


<img width="852" height="626" alt="image" src="https://github.com/user-attachments/assets/654f15cb-dda4-43a0-bf9a-c82ea931baf6" />

```
export TMPDIR=/dev/shm; python NorMuon_hybrid.logging.optuna.NaN-safe.py --output_dir /16optane/model_muon_qat_checkpoints --fp16 --resume_from /16optane/model_muon_qat_checkpoints/ --disable_qat  --save_steps 20  --learning_rate 1e-7  --use_optuna --optuna_trials 30 --optuna_dataset_samples 100 --optuna_nan_retries 2 --attention_backend sdpa --batch_size 1 --optimizer_8bit
 
Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.0+cu128).
📥 Loading data from /content/training_data_19century.json...
✅ Loaded 43960 examples
📦 Loading model oopere/gemma-3-270m-14L-distilled...
Loading weights: 100%|██████| 158/158 [00:00<00:00, 3982.36it/s]
The module name  (originally ) is not a valid Python identifier. Please rename the original module to avoid import issues.
✅ Model loaded. Parameters: 234.66M

⏩ Skipping benchmarks
Map: 100%|██████| 100/100 [00:00<00:00, 5159.05 examples/s]
📦 Pre-tokenized dataset for Optuna: 100 examples
[I 2026-06-11 11:35:18,833] Using an existing study with name 'noromuon_hybrid' instead of creating a new one.

🔬 Trial 70: bs=4, ga=1, lr=2.19e-06, warmup=0.109, grad_clip=0.67, sched=cosine
...


🏆 Best parameters found: {'batch_size': 1, 'grad_accum': 1, 'learning_rate': 4.859854023651697e-06, 'warmup_ratio': 0.10305464809237566, 'max_grad_norm': 1.6124260665977124, 'lr_scheduler': 'cosine'}
✅ Optuna search completed. Using batch_size=1, grad_accum=1, lr=4.86e-06, warmup=0.10305464809237566, max_grad_norm=1.6124260665977124, scheduler=cosine
🔧 Using 8-bit AdamW optimizer (lr=4.859854023651697e-06)
 
 67%|███ ▋                                             | 10/15 [00:36<00:18,  3.68s/it]
Adding EOS to train dataset: 100%|████ | 43960/43960 [00:02<00:00, 17170.35 examples/s]
Tokenizing train dataset: 100%|████ | 43960/43960 [00:15<00:00, 2869.54 examples/s]

🚀 Starting training...
  0%|                                                                                                                                 | 10/131880 [00:17<63:05:45,  1.72s/it]step 10/1000000 │ loss 3.6551 │ ema 3.6551 │ lr 1.79e-09 │ gnorm 202.35 │       95 tok/s │ VRAM 0.26 GB free │ acc 0.455 │ entropy 1.280
```
Accuracy  0.645:
```
rg -A9999 "3.655.*entropy 1.280" pretrain.log|sort -Vk25|tail|column -t

step  10850/1000000  │  loss  1.9946  │  ema  1.9946  │  lr  3.88e-06  │  gnorm  30.30  │  232  tok/s  │  VRAM  0.30  GB  free  │  acc  0.605  │  entropy  2.135
step  6980/1000000   │  loss  1.9726  │  ema  1.9726  │  lr  2.49e-06  │  gnorm  37.79  │  232  tok/s  │  VRAM  2.33  GB  free  │  acc  0.607  │  entropy  1.965
step  2850/1000000   │  loss  1.9644  │  ema  1.9644  │  lr  1.02e-06  │  gnorm  56.74  │  236  tok/s  │  VRAM  0.88  GB  free  │  acc  0.607  │  entropy  2.158
step  7360/1000000   │  loss  1.8794  │  ema  1.8794  │  lr  2.63e-06  │  gnorm  49.79  │  233  tok/s  │  VRAM  2.33  GB  free  │  acc  0.610  │  entropy  1.998
step  2700/1000000   │  loss  1.8828  │  ema  1.8828  │  lr  9.63e-07  │  gnorm  62.55  │  236  tok/s  │  VRAM  2.33  GB  free  │  acc  0.611  │  entropy  1.993
step  8190/1000000   │  loss  1.9476  │  ema  1.9476  │  lr  2.93e-06  │  gnorm  31.73  │  232  tok/s  │  VRAM  0.88  GB  free  │  acc  0.620  │  entropy  2.163
step  7770/1000000   │  loss  1.7820  │  ema  1.7820  │  lr  2.78e-06  │  gnorm  42.57  │  233  tok/s  │  VRAM  0.89  GB  free  │  acc  0.622  │  entropy  1.942
step  7740/1000000   │  loss  1.8532  │  ema  1.8532  │  lr  2.76e-06  │  gnorm  48.15  │  233  tok/s  │  VRAM  2.33  GB  free  │  acc  0.625  │  entropy  1.821
step  10140/1000000  │  loss  1.7915  │  ema  1.7915  │  lr  3.62e-06  │  gnorm  44.31  │  231  tok/s  │  VRAM  2.33  GB  free  │  acc  0.631  │  entropy  1.946
step  6280/1000000   │  loss  1.5950  │  ema  1.5950  │  lr  2.24e-06  │  gnorm  48.28  │  233  tok/s  │  VRAM  2.33  GB  free  │  acc  0.645  │  entropy  1.831
```

loss  1.5950:
```
rg -A9999 "3.655.*entropy 1.280" pretrain.log|awk '{print $8,$0}'|sort -rVk1|column -t|tail
1.9644  step  2850/1000000   │  loss  1.9644  │  ema  1.9644  │  lr  1.02e-06  │  gnorm  56.74   │  236  tok/s  │  VRAM  0.88  GB  free  │  acc  0.607  │  entropy  2.158
1.9608  step  9920/1000000   │  loss  1.9608  │  ema  1.9608  │  lr  3.54e-06  │  gnorm  55.45   │  231  tok/s  │  VRAM  2.33  GB  free  │  acc  0.563  │  entropy  1.968
1.9476  step  8190/1000000   │  loss  1.9476  │  ema  1.9476  │  lr  2.93e-06  │  gnorm  31.73   │  232  tok/s  │  VRAM  0.88  GB  free  │  acc  0.620  │  entropy  2.163
1.8905  step  9060/1000000   │  loss  1.8905  │  ema  1.8905  │  lr  3.24e-06  │  gnorm  30.03   │  232  tok/s  │  VRAM  2.33  GB  free  │  acc  0.603  │  entropy  1.952
1.8828  step  2700/1000000   │  loss  1.8828  │  ema  1.8828  │  lr  9.63e-07  │  gnorm  62.55   │  236  tok/s  │  VRAM  2.33  GB  free  │  acc  0.611  │  entropy  1.993
1.8794  step  7360/1000000   │  loss  1.8794  │  ema  1.8794  │  lr  2.63e-06  │  gnorm  49.79   │  233  tok/s  │  VRAM  2.33  GB  free  │  acc  0.610  │  entropy  1.998
1.8532  step  7740/1000000   │  loss  1.8532  │  ema  1.8532  │  lr  2.76e-06  │  gnorm  48.15   │  233  tok/s  │  VRAM  2.33  GB  free  │  acc  0.625  │  entropy  1.821
1.7915  step  10140/1000000  │  loss  1.7915  │  ema  1.7915  │  lr  3.62e-06  │  gnorm  44.31   │  231  tok/s  │  VRAM  2.33  GB  free  │  acc  0.631  │  entropy  1.946
1.7820  step  7770/1000000   │  loss  1.7820  │  ema  1.7820  │  lr  2.78e-06  │  gnorm  42.57   │  233  tok/s  │  VRAM  0.89  GB  free  │  acc  0.622  │  entropy  1.942
1.5950  step  6280/1000000   │  loss  1.5950  │  ema  1.5950  │  lr  2.24e-06  │  gnorm  48.28   │  233  tok/s  │  VRAM  2.33  GB  free  │  acc  0.645  │  entropy  1.831
```
