from src.logger import logger
from datasets import Dataset, load_metric
import torch
import time

class PerformanceBenchmark:
    ## TODO: Add naive baseline
    ## TODO: use evaluate module from huggingface
    ## TODO: add tqdm
    def __init__(self, model, tokenizer, tuning_name:str, dataloader, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.tuning_name = tuning_name
        self.device = device

    def get_preds_labels(self):

        with torch.no_grad():
            preds, labels = [], []
            self.model.to(self.device)
            for example in self.dataloader:
                pred = self.model(example['input_ids'].to(self.device), attention_mask=example['attention_mask'].to(self.device))
                pred = pred.logits.argmax(dim=1).tolist()
                label = example['label'].tolist()
                preds.extend(pred)
                labels.extend(label)

        return preds, labels

    def compute_precision(self, preds:list, labels:list):
        precision = load_metric('precision')
        ## TODO: look up the different precisions
        precision_score = precision.compute(predictions=preds, references=labels, average='macro')

        return precision_score['precision']
    
    def compute_f1(self, preds:list, labels:list):
        f1 = load_metric('f1')
        f1_score = f1.compute(predictions=preds, references=labels, average='macro')

        return f1_score['f1']
    
    def run_benchmark(self):
        preds, labels = self.get_preds_labels()
        metrics = {}
        metrics[self.tuning_name] = {}
        metrics[self.tuning_name]['precision'] = self.compute_precision(preds, labels)
        metrics[self.tuning_name]['f1'] = self.compute_f1(preds, labels)
        return metrics
    
def get_performance_benchmark_test(model, tokenizer, device, dataloader, test_name:str):
    # TEST: 1k samples; doing this with the mps device is 4.4x faster than the cpu
    start_time = time.perf_counter()
    pb = PerformanceBenchmark(model, tokenizer, test_name, dataloader, device=device) 
    metrics = pb.run_benchmark()
    end_time = time.perf_counter()
    logger.info(f'Finished {test_name} run in {end_time - start_time} seconds')

    logger.info(metrics)
    return metrics