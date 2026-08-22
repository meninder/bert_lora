#!/bin/bash
source /Users/meninderpurewal/miniconda3/bin/activate hf2
python lora_timing_script.py --run_name base_8 
python lora_timing_script.py --run_name grad_chkpt_8 --grad_chkpt 
python lora_timing_script.py --run_name grad_chkpt_16 --grad_chkpt --batch_size 16 --epochs 4
python lora_timing_script.py --run_name grad_chkpt_acc --grad_chkpt --grad_acc_steps 2 --epochs 4
python lora_timing_script.py --run_name base_8_l11 --lora_layers 11 
python lora_timing_script.py --run_name base_8_l1 --lora_layers 1 
python lora_timing_script.py --run_name adafactor --optim adafactor 
python lora_timing_script.py --run_name adamw_hf --optim adamw_hf 

