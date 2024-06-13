from datasets import load_dataset
from transformers import AutoTokenizer
import torch

def preprocess():
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    raw_datasets = load_dataset("glue", "mrpc")

    # 데이터셋 전처리
    train_data = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'labels': []}
    for i in range(len(raw_datasets['train'])):
        tokenize = tokenizer(raw_datasets['train']['sentence1'][i], raw_datasets['train']['sentence2'][i], truncation=True)
        train_data['input_ids'].append(tokenize['input_ids'])
        train_data['token_type_ids'].append(tokenize['token_type_ids'])
        train_data['attention_mask'].append(tokenize['attention_mask'])
        train_data['labels'].append(raw_datasets['train']['label'][i])

    valid_data = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'labels': []}
    for i in range(len(raw_datasets['validation'])):
        tokenize = tokenizer(raw_datasets['validation']['sentence1'][i], raw_datasets['validation']['sentence2'][i], truncation=True)
        valid_data['input_ids'].append(tokenize['input_ids'])
        valid_data['token_type_ids'].append(tokenize['token_type_ids'])
        valid_data['attention_mask'].append(tokenize['attention_mask'])
        valid_data['labels'].append(raw_datasets['validation']['label'][i])

    torch.save(train_data, '/app/train_data.pt')
    torch.save(valid_data, '/app/valid_data.pt')

if __name__ == "__main__":
    preprocess()
