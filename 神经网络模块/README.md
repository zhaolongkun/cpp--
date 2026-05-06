# Neural Module For PNR-IMM-KF

This folder contains the neural-network module design and training pipeline
for the pixel-domain filter in this project.

## Goal
Train a sequence model that outputs adaptive filter parameters:

1. `bias_x` (px)
2. `bias_y` (px)
3. `alpha_q` (process noise scale)
4. `alpha_r` (measurement noise scale)
5. `outlier_prob` (0~1)

These outputs are designed to be embedded into the Kalman/IMM update loop.

## Training Policy (Strict)
- No pretrained model is used.
- Weights are randomly initialized and trained from scratch on your task data.
- The checkpoint metadata records:
  `training_mode=from_scratch_random_init`, `pretrained_weights=false`.

## Files
- `data_utils.py`: CSV parsing, pseudo-label generation, sequence builder
- `build_dataset.py`: generate train/val dataset `.npz`
- `model.py`: GRU model (`NCEGRU`)
- `train_nce.py`: training script
- `export_onnx.py`: export best checkpoint to ONNX
- `infer_onnx.py`: run ONNX inference on tracker CSV
- `evaluate_log.py`: compute paper metrics from tracker CSV
- `compare_metrics.py`: compare two metric json files
- `collect_and_eval.bat`: stage-1 run (collect data + evaluate)
- `requirements.txt`: Python dependencies

## Input CSV
Default input:
`..\logs\tracker_log.csv`

Expected columns include:
`x_error,y_error,dx_hat,dy_hat,cmd_x,cmd_y,bbox_x1,bbox_y1,bbox_x2,bbox_y2,det_count,track_count,coast_count,is_meas_update,meas_age_ms,pnr_gate_d2,pnr_model_cv,pnr_model_ca,note`

## Quick Start (Windows)
```bat
cd /d D:\kun-data\kun-code-data\反无\cpp智能控制\神经网络模块
conda activate py310
python -m pip install -r requirements.txt

python build_dataset.py ^
  --csv "..\logs\tracker_log.csv" ^
  --out ".\artifacts\dataset.npz" ^
  --seq_len 12

python train_nce.py ^
  --dataset ".\artifacts\dataset.npz" ^
  --out_dir ".\artifacts\runs\exp1" ^
  --epochs 40 ^
  --batch_size 256

python export_onnx.py ^
  --ckpt ".\artifacts\runs\exp1\best.pt" ^
  --out ".\artifacts\runs\exp1\nce_pnr.onnx"

python infer_onnx.py ^
  --onnx ".\artifacts\runs\exp1\nce_pnr.onnx" ^
  --meta ".\artifacts\runs\exp1\nce_pnr.meta.json" ^
  --csv "..\logs\tracker_log.csv" ^
  --out_csv ".\artifacts\runs\exp1\onnx_pred.csv"
```

Or run pipeline directly:
```bat
train_pipeline.bat
```

## Stage-1 (No Data Yet)
Run robust IMM-KF only (no neural) to collect your own dataset:
```bat
cd /d D:\kun-data\kun-code-data\反无\cpp智能控制\神经网络模块
collect_and_eval.bat
```

The script will:
1. build C++ tracker
2. run camera mode
3. save session log to `datasets/`
4. export metrics json to `metrics/`

Then start stage-2 training from scratch on your collected data.

## Output Order
Model output tensor shape: `[B, 5]` with order:
`[bias_x, bias_y, alpha_q, alpha_r, outlier_prob]`

## Notes
- Labels are pseudo-labels generated from robust trend/noise statistics.
- This is a practical training baseline for the integrated neural-Kalman loop.
- For publication, replace pseudo-labels with stronger supervision if available
  (GT trajectories, synchronized gimbal states, and target angle annotations).
- Recommended runtime: Anaconda `py310` environment.
- Default runtime config now uses `filter.neural_enable=false` (stage-1 safe mode).
