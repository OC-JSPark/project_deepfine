import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AdamW, get_scheduler
from transformers import DataCollatorWithPadding, AutoTokenizer
from tqdm.auto import tqdm

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, index):
        return {k: torch.tensor(v[index]) for k, v in self.data.items()}

def train():
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    train_data = torch.load('/app/train_data.pt')
    valid_data = torch.load('/app/valid_data.pt')

    train_dataset = MyDataset(train_data)
    eval_dataset = MyDataset(valid_data)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8, collate_fn=data_collator)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    torch.save(model.state_dict(), '/app/model.pth')

if __name__ == "__main__":
    train()
