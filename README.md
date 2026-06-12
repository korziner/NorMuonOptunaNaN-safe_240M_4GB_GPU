# NorMuonOptunaNaN-safe_240M_4GB_GPU
Continued full pretrain for gemma3 270M on 4GB GPU

--attention_backend {eager,sdpa}  Бэкенд внимания (по умолч. eager)
sdpa supported by gemma3
<img width="1175" height="703" alt="image" src="https://github.com/user-attachments/assets/40270859-cbe0-4bdd-aa29-ec20d1e0d1a8" />


<img width="852" height="626" alt="image" src="https://github.com/user-attachments/assets/654f15cb-dda4-43a0-bf9a-c82ea931baf6" />

```
export TMPDIR=/dev/shm; python NorMuon_hybrid.logging.optuna.NaN-safe.py --output_dir /16optane/model_muon_qat_checkpoints --fp16 --resume_from /16optane/model_muon_qat_checkpoints/ --disable_qat  --save_steps 20  --learning_rate 1e-7  --use_optuna --optuna_trials 30 --optuna_dataset_samples 100 --optuna_nan_retries 2 --attention_backend sdpa --batch_size 1 --optimizer_8bit
 
Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.0+cu128).
📥 Loading data from /content/training_data_19century.json...
✅ Loaded 43960 examples
📦 Loading model oopere/gemma-3-270m-14L-distilled...
Loading weights: 100%|██████| 158/158 [00:00<00:00, 3982.36it/s]
 
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
Accuracy  0.746:
```tail -500 pretrain.log|sort -Vk25|tail|column -t
step  13970/1000000  │  loss  1.2720  │  ema  1.2720  │  lr  4.86e-06  │  gnorm  35.03  │  249  tok/s  │  VRAM  0.13  GB  free  │  acc  0.720  │  entropy  1.377
step  13350/1000000  │  loss  1.2952  │  ema  1.2952  │  lr  4.77e-06  │  gnorm  36.76  │  249  tok/s  │  VRAM  0.27  GB  free  │  acc  0.720  │  entropy  1.383
step  13100/1000000  │  loss  1.2860  │  ema  1.2860  │  lr  4.68e-06  │  gnorm  39.88  │  249  tok/s  │  VRAM  2.35  GB  free  │  acc  0.722  │  entropy  1.286
step  12200/1000000  │  loss  1.2332  │  ema  1.2332  │  lr  4.36e-06  │  gnorm  50.46  │  249  tok/s  │  VRAM  2.35  GB  free  │  acc  0.725  │  entropy  1.367
step  11770/1000000  │  loss  1.2523  │  ema  1.2523  │  lr  4.21e-06  │  gnorm  39.00  │  249  tok/s  │  VRAM  0.06  GB  free  │  acc  0.726  │  entropy  1.301
step  13040/1000000  │  loss  1.2527  │  ema  1.2527  │  lr  4.66e-06  │  gnorm  55.85  │  249  tok/s  │  VRAM  0.27  GB  free  │  acc  0.726  │  entropy  1.355
step  13900/1000000  │  loss  1.2662  │  ema  1.2662  │  lr  4.86e-06  │  gnorm  32.73  │  249  tok/s  │  VRAM  2.35  GB  free  │  acc  0.730  │  entropy  1.373
step  11820/1000000  │  loss  1.3078  │  ema  1.3078  │  lr  4.22e-06  │  gnorm  47.37  │  249  tok/s  │  VRAM  0.27  GB  free  │  acc  0.733  │  entropy  1.312
step  14020/1000000  │  loss  1.1704  │  ema  1.1704  │  lr  4.86e-06  │  gnorm  45.48  │  249  tok/s  │  VRAM  0.29  GB  free  │  acc  0.737  │  entropy  1.199
step  13840/1000000  │  loss  1.1426  │  ema  1.1426  │  lr  4.86e-06  │  gnorm  46.68  │  249  tok/s  │  VRAM  0.25  GB  free  │  acc  0.746  │  entropy  1.231
```

loss  1.1426:
```
tail -500 pretrain.log|awk '{print $8,$0}'|sort -rVk1|column -t|tail
1.2660  step  11640/1000000  │  loss  1.2660  │  ema  1.2660  │  lr  4.16e-06  │  gnorm  41.27  │  249  tok/s  │  VRAM  0.25  GB  free  │  acc  0.699  │  entropy  1.422
1.2596  step  11870/1000000  │  loss  1.2596  │  ema  1.2596  │  lr  4.24e-06  │  gnorm  31.61  │  249  tok/s  │  VRAM  0.19  GB  free  │  acc  0.704  │  entropy  1.398
1.2527  step  13040/1000000  │  loss  1.2527  │  ema  1.2527  │  lr  4.66e-06  │  gnorm  55.85  │  249  tok/s  │  VRAM  0.27  GB  free  │  acc  0.726  │  entropy  1.355
1.2523  step  11770/1000000  │  loss  1.2523  │  ema  1.2523  │  lr  4.21e-06  │  gnorm  39.00  │  249  tok/s  │  VRAM  0.06  GB  free  │  acc  0.726  │  entropy  1.301
1.2475  step  13380/1000000  │  loss  1.2475  │  ema  1.2475  │  lr  4.78e-06  │  gnorm  37.63  │  249  tok/s  │  VRAM  0.27  GB  free  │  acc  0.711  │  entropy  1.312
1.2332  step  12200/1000000  │  loss  1.2332  │  ema  1.2332  │  lr  4.36e-06  │  gnorm  50.46  │  249  tok/s  │  VRAM  2.35  GB  free  │  acc  0.725  │  entropy  1.367
1.2251  step  14050/1000000  │  loss  1.2251  │  ema  1.2251  │  lr  4.86e-06  │  gnorm  35.84  │  249  tok/s  │  VRAM  0.29  GB  free  │  acc  0.720  │  entropy  1.333
1.2155  step  11670/1000000  │  loss  1.2155  │  ema  1.2155  │  lr  4.17e-06  │  gnorm  41.48  │  249  tok/s  │  VRAM  0.25  GB  free  │  acc  0.708  │  entropy  1.325
1.1704  step  14020/1000000  │  loss  1.1704  │  ema  1.1704  │  lr  4.86e-06  │  gnorm  45.48  │  249  tok/s  │  VRAM  0.29  GB  free  │  acc  0.737  │  entropy  1.199
1.1426  step  13840/1000000  │  loss  1.1426  │  ema  1.1426  │  lr  4.86e-06  │  gnorm  46.68  │  249  tok/s  │  VRAM  0.25  GB  free  │  acc  0.746  │  entropy  1.231
```
<img width="1761" height="1031" alt="image" src="https://github.com/user-attachments/assets/9c12289d-cb0e-4c7d-b2e1-732ed1b85dab" />
