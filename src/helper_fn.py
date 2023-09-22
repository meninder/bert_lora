from datasets import load_dataset
from torch.utils.data import DataLoader
from src.logger import logger
import re


from transformers import BertForSequenceClassification, BertTokenizer
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model
import evaluate
import numpy as np


    

def get_dataset(tokenizer, cap=None):
    dataset = load_dataset("tweet_eval", name="sentiment")
    dataset_tokenized = dataset.map(clean_text).map(
        lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length")
        , batched=True)
    train_dataset = dataset_tokenized["train"].with_format("torch")
    test_dataset = dataset_tokenized["test"].with_format("torch")
    logger.info(f"Train shape: {train_dataset.shape}, Test shape: {test_dataset.shape}")
    if cap:
        logger.info(f"Capping")
        train_dataset = train_dataset.select(range(cap))
        test_dataset = test_dataset.select(range(cap))
        logger.info(f"Train shape: {train_dataset.shape}, Test shape: {test_dataset.shape}")

    return train_dataset, test_dataset

def get_dataloader(train_dataset, test_dataset, batch_size=16):
    train_dataset = train_dataset.select_columns(['input_ids', 'label', 'attention_mask'])
    test_dataset = test_dataset.select_columns(['input_ids', 'label', 'attention_mask'])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def clean_text(sample):
    def remove_unicode_regex(text):
        # Define the regex pattern to match Unicode characters
        pattern = r"[\u0080-\uFFFF]"

        # Use regex substitution to remove the matched Unicode characters
        cleaned_text = re.sub(pattern, "", text)

        return cleaned_text

    sample["text"] = sample["text"].replace("@user", "")
    sample["text"] = sample["text"].replace("#", "")
    sample["text"] = sample["text"].replace("&amp;", "&")
    sample["text"] = sample["text"].replace("&lt;", "<")
    sample["text"] = sample["text"].replace("&gt;", ">")
    sample["text"] = sample["text"].strip()
    sample["text"] = remove_unicode_regex(sample["text"])

    return sample


def get_model_for_training(fine_tuning:str):
    # confirm that fine_tuning variable is one of the following ['no', 'top', 'top2', 'lora']
    choices = ['no', 'top', 'top2', 'lora']
    assert fine_tuning in choices, "fine_tuning variable must be one of ['no', 'top', 'top2', 'lora']"

    
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=3)
    
    if fine_tuning == 'top':
        # freeze all but the final encoder
        last_layer = model.bert.encoder.layer[-1]
        for param in model.bert.parameters():
            param.requires_grad = False

        for param in last_layer.parameters():
            param.requires_grad = True

        logger.info(f"Last layer parameter count: {sum(p.numel() for p in last_layer.parameters())}")
    
    elif fine_tuning == 'top2':
        last_two_layers = model.bert.encoder.layer[-2:]
        for param in model.bert.parameters():
            param.requires_grad = False

        for param in last_two_layers.parameters():
            param.requires_grad = True

        logger.info(f"Last two layer parameter count: {sum(p.numel() for p in last_two_layers.parameters())}")

    elif fine_tuning == 'lora':
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=4,
            lora_alpha=1,
            lora_dropout=0.1,
        )
        model = get_peft_model(model, lora_config)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        num_trainable_params = sum(p.numel() for p in trainable_params)
        logger.info(f"LoRA parameter count: {num_trainable_params}")

    return model, tokenizer


def get_trainer(fine_tuning, output_dir, epochs=1, batch_size=16, device='mps', cap=None):
    model, tokenizer = get_model_for_training(fine_tuning=fine_tuning)
    train_dataset, test_dataset = get_dataset(tokenizer, cap=cap)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        evaluation_strategy="steps",
        save_steps=1_000,
        eval_steps=1_000,
        save_total_limit=3,
        use_mps_device= device=='mps',
        logging_dir=output_dir + '/logs',
        )
    precision_metric = evaluate.load("precision")
    f1_metric = evaluate.load("f1")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=lambda pred: {
            "precision": precision_metric.compute(
                predictions=np.argmax(pred.predictions, axis=-1),
                references=pred.label_ids,
                average="macro",
            )["precision"],
            "f1": f1_metric.compute(
                predictions=np.argmax(pred.predictions, axis=-1),
                references=pred.label_ids,
                average="macro",
            )["f1"],
        },
    )

    return trainer, model

def train_save_evaluate(trainer, fine_tuning_name:str, output_dir, model):
    
    if fine_tuning_name != 'no':
        logger.info(f'*****Fine-tuning {fine_tuning_name}*****')
        trainer.train()
    if fine_tuning_name =="lora":
        logger.info(f'*****Saving {fine_tuning_name}*****')
        model.save_pretrained(output_dir+'/lora_model')
    else:
        trainer.save_model(output_dir+'/'+fine_tuning_name)

    logger.info(f'*****Evaluating on {fine_tuning_name}*****')
    results = trainer.evaluate()
    trainer.save_metrics(f"eval", results)

    return results


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )