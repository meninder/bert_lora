import sys
import os
import wandb
import argparse
from src.helper_fn import get_trainer, train_save_evaluate
from src.logger import logger
from dotenv import load_dotenv

def main(args):
    '''

    '''

    load_dotenv()
    wandb_project = os.environ['WANDB_PROJECT']
    logger.info(f'***** WandB Project: {wandb_project} *****')
    parser = argparse.ArgumentParser()

    # I dont anticipate changing these
    parser.add_argument('--fine_tuning_name', type=str, default='lora')
    parser.add_argument('--output_dir', type=str, default='lora_timing_scripts')
    parser.add_argument('--device', type=str, default='mps')

    # I might change these, but not often
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--cap_rows_train', type=int, default=10_000)
    parser.add_argument('--cap_rows_test', type=int, default=1_000)

    # These will change
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--run_name', type=str, default='default')
    #opt choices: adamw_hf, adamw_torch, adamw_torch_fused, adamw_apex_fused, adamw_anyprecision, adafactor.
    parser.add_argument('--optim', type=str, default='adamw_torch')
    parser.add_argument('--grad_acc_steps', type=int, default=1)
    parser.add_argument('--grad_chkpt', action='store_true', default=False)
    parser.add_argument('--fp16_bool', action='store_true', default=False)
    parser.add_argument('--lora_layers', nargs='+', type=int, default=None)

    args = parser.parse_args(args)

    logger.info(f'*****Running with the following args {args}*****')
    trainer = get_trainer(fine_tuning_name=args.fine_tuning_name, 
                                 output_dir=args.output_dir, 
                                 epochs=args.epochs, 
                                 batch_size=args.batch_size, 
                                 device=args.device, 
                                 cap_rows=True,
                                 cap_rows_train=args.cap_rows_train, 
                                 cap_rows_test=args.cap_rows_test, 
                                 run_name=args.run_name,
                                 optim=args.optim,
                                 grad_acc_steps=args.grad_acc_steps,
                                 grad_chkpt=args.grad_chkpt,
                                 fp16_bool=args.fp16_bool,
                                 lora_layers=args.lora_layers)
    
    logger.info(f'*****Fine-tuning {args.run_name}*****')
    trainer.train()
    logger.info(f'*****Evaluating on {args.run_name}*****')
    trainer.evaluate()
    logger.info(f'*****Ending wandb instance*****')
    wandb.finish()


if __name__ == '__main__':
    main(sys.argv[1:])
