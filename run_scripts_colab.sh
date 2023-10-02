python lora_timing_script.py --run_name base_8 --device cuda
python lora_timing_script.py --run_name grad_chkpt_8 --device cuda --grad_chkpt 
python lora_timing_script.py --run_name grad_chkpt_16 --device cuda --grad_chkpt --batch_size 16 --epochs 4
python lora_timing_script.py --run_name grad_chkpt_acc --device cuda --grad_chkpt --grad_acc_steps 2 --epochs 4
python lora_timing_script.py --run_name base_8_l11 --lora_layers 11 --device cuda
python lora_timing_script.py --run_name base_8_l1 --lora_layers 1 --device cuda
python lora_timing_script.py --run_name adafactor --optim adafactor --device cuda
python lora_timing_script.py --run_name adamw_hf --optim adamw_hf --device cuda

