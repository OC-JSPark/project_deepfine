import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, index):
        return {k: torch.tensor(v[index]) for k, v in self.data.items()}

def test():
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    valid_data = torch.load('/app/valid_data.pt')

    eval_dataset = MyDataset(valid_data)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8, collate_fn=data_collator)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    model.load_state_dict(torch.load('/app/model.pth'))
    model.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    accuracy = 0
    f1 = 0

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        acc = accuracy_score(batch['labels'].cpu(), predictions.cpu())
        f1 += f1_score(batch['labels'].cpu(), predictions.cpu(), average='weighted')
        accuracy += acc

    accuracy /= len(eval_dataloader)
    f1 /= len(eval_dataloader)

    with open('/app/metrics.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy}\nF1_Score: {f1}")

if __name__ == "__main__":
    test()
