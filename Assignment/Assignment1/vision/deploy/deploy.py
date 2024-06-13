import torch
from transformers import ViTForImageClassification

def deploy():
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=10)
    model.load_state_dict(torch.load('/app/models/vit_model.pth'))
    torch.save(model, '/app/models/deployed_model.pth')

if __name__ == "__main__":
    deploy()
