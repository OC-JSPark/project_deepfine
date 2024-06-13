import torch
from transformers import ViTForImageClassification, AutoImageProcessor
import numpy as np
import evaluate
from torch.utils.data import DataLoader
from torchvision import transforms

def collator(data, transform):
    images, labels = zip(*data)
    pixel_values = torch.stack([transform(image) for image in images])
    labels = torch.tensor([label for label in labels])
    return {"pixel_values": pixel_values, "labels": labels}

def test():
    subset_test_dataset = torch.load('/app/subset_test_dataset.pt')

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_processor.size['height'], image_processor.size['width'])),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])

    valid_dataloader = DataLoader(
        subset_test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: collator(x, transform),
        drop_last=True
    )

    classes = subset_test_dataset.dataset.classes
    class_to_idx = subset_test_dataset.dataset.class_to_idx

    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=len(classes))
    model.load_state_dict(torch.load('/app/models/vit_model.pth'))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    metric = evaluate.load('f1')
    y_true = []
    y_pred = []

    for batch in valid_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        y_true.extend(batch['labels'].cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    macro_f1 = metric.compute(predictions=y_pred, references=y_true, average="macro")
    accuracy = (y_true == y_pred).mean()

    with open('/app/metrics.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy}\nF1_Score: {macro_f1['f1']}")

if __name__ == "__main__":
    test()
