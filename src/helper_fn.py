from datasets import load_dataset
from torch.utils.data import DataLoader
from src.logger import logger
import re


from transformers import BertForSequenceClassification, BertTokenizer
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model
import evaluate
import numpy as np
import wandb


    

def get_dataset(tokenizer:BertTokenizer, 
                dataset_name:str='yelp_polarity', 
                cap_rows:bool=True, 
                cap_rows_train:int=10_000, 
                cap_rows_test:int=2_000):
    
    dataset = load_dataset(dataset_name)
    if cap_rows:
        logger.info(f"Capping dataset rows to {cap_rows_train} train and {cap_rows_test} test")
        train_dataset = dataset["train"]
        train_dataset = train_dataset.select(np.random.randint(0, len(train_dataset), cap_rows_train))
        test_dataset = dataset["test"]
        test_dataset = test_dataset.select(np.random.randint(0, len(test_dataset), cap_rows_test))

    train_dataset_tokenized = train_dataset.map(
        lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length")
            , batched=True)
    test_dataset_tokenized = test_dataset.map(
        lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length")
            , batched=True)

    logger.info(f"Train shape: {train_dataset.shape}, Test shape: {test_dataset.shape}")
    train_dataset_tokenized=train_dataset_tokenized.with_format("torch", )
    test_dataset_tokenized=test_dataset_tokenized.with_format("torch", )

    return train_dataset_tokenized, test_dataset_tokenized

def get_dataloader(train_dataset, test_dataset, batch_size=16):
    train_dataset = train_dataset.select_columns(['input_ids', 'label', 'attention_mask'])
    test_dataset = test_dataset.select_columns(['input_ids', 'label', 'attention_mask'])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def get_model_for_training(fine_tuning:str, lora_layers:list=None):
    
    choices = ['none', 'full', 'top', 'top2', 'lora', 'classifier']
    assert fine_tuning in choices, "fine_tuning variable must be one of the following: {choices}"

    
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)
    
    if fine_tuning == 'none':
        # freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        logger.info(f"Training no layers")
        
    elif fine_tuning == 'classifier':
        # freeze all but classifier
        classifier = model.classifier
        for param in model.bert.parameters():
            param.requires_grad = False
        for param in classifier.parameters():
            param.requires_grad = True
        logger.info(f"Training only classifier")

    elif fine_tuning == 'top':
        # freeze all but the final encoder and classifier
        last_layer = model.bert.encoder.layer[-1]
        classifier = model.classifier
        for param in model.bert.parameters():
            param.requires_grad = False
        for param in classifier.parameters():
            param.requires_grad = True

        for param in last_layer.parameters():
            param.requires_grad = True

        logger.info(f"Training only last layer of encoder + classifier")
    
    elif fine_tuning == 'top2':
        last_two_layers = model.bert.encoder.layer[-2:]
        classifier = model.classifier
        for param in model.bert.parameters():
            param.requires_grad = False

        for param in last_two_layers.parameters():
            param.requires_grad = True
        for param in classifier.parameters():
            param.requires_grad = True
        
        logger.info(f"Training only last two layers of encoder + classifier")

    elif fine_tuning == 'lora':
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=4,
            lora_alpha=1,
            lora_dropout=0.1,
            layers_to_transform=lora_layers,
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"Training LoRA")
    else:
        logger.info(f"Training full model")

    logger.info(get_trainable_parameters(model))
    return model, tokenizer

def compute_metrics(eval_pred):
    precision_metric = evaluate.load("precision")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")
    
    preds = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids

    metric = {}
    metric.update(precision_metric.compute(predictions=preds, references=labels, average="macro"))
    metric.update(f1_metric.compute(predictions=preds, references=labels, average="macro"))
    metric.update(accuracy_metric.compute(predictions=preds, references=labels) )

    return metric


def get_trainer(fine_tuning_name:str, # change to choice
                output_dir:str, # change to path
                epochs=2, 
                batch_size=16, 
                device='mps', 
                cap_rows=True,
                cap_rows_train=10_000,
                cap_rows_test=2_000,
                run_name='default',
                optim='adamw_torch',
                grad_acc_steps=1,
                grad_chkpt=False,
                fp16_bool=False,
                lora_layers=None):
    
    logger.info('*****Getting Model and Tokenizer*****')
    model, tokenizer = get_model_for_training(fine_tuning=fine_tuning_name, lora_layers=lora_layers)
    logger.info('*****Getting Dataset*****')
    train_dataset, test_dataset = get_dataset(tokenizer, cap_rows=cap_rows, cap_rows_train=cap_rows_train, 
    cap_rows_test=cap_rows_test)
    
    eval_steps = int(epochs * len(train_dataset) // batch_size / 4 ) 
    logger.info(f'Calculated eval_steps: {eval_steps}')

    if hasattr(model, "enable_input_require_grads"):
        logger.info(f"*****Model input require grads enabled*****")
        model.enable_input_require_grads()
        

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        evaluation_strategy="steps",
        save_strategy="no",
        #save_steps=800,
        eval_steps=eval_steps,
        save_total_limit=1,
        use_mps_device= device=='mps',
        logging_dir=output_dir + '/logs',
        report_to='wandb',
        run_name=run_name,
        optim=optim,
        gradient_accumulation_steps=grad_acc_steps,
        gradient_checkpointing=grad_chkpt,
        fp16=fp16_bool
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    return trainer, model

def train_save_evaluate(trainer, fine_tuning_name:str, output_dir, model):
    if fine_tuning_name != 'none':
        logger.info(f'*****Fine-tuning {fine_tuning_name}*****')
        trainer.train()
    else:
        logger.info(f'*****Skipping fine-tuning for tuning type {fine_tuning_name}*****')
    
    logger.info(f'*****Saving {fine_tuning_name} in {output_dir}/{fine_tuning_name}*****')
    if fine_tuning_name =="lora":
        model.save_pretrained(output_dir+'/'+fine_tuning_name)
    elif fine_tuning_name != 'none':
        trainer.save_model(output_dir+'/'+fine_tuning_name)
    else:
        logger.info(f'*****Skipping saving for tuning type {fine_tuning_name}*****')

    logger.info(f'*****Evaluating on {fine_tuning_name}*****')
    results = trainer.evaluate()
    trainer.save_metrics("eval", results)

    logger.info(f'*****Ending wandb instance*****')
    wandb.finish()
    return results


def get_trainable_parameters(model):
    """
    Gets the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
