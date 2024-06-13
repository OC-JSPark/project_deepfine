import torch
from transformers import AutoModelForSequenceClassification

def deploy():
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.load_state_dict(torch.load('/app/model.pth'))
    torch.save(model, '/app/deployed_model.pth')

if __name__ == "__main__":
    deploy()
