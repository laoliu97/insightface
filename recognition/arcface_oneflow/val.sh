#!/bin/bash
export ONEFLOW_DEBUG_MODE=True
python val.py configs/ms1mv3_r50 --model_path work_dirs/webface600k_r50/epoch_19
