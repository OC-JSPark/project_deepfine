import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
import os

def collator(data, transform):
    images, labels = zip(*data)
    pixel_values = torch.stack([transform(image) for image in images])
    labels = torch.tensor([label for label in labels])
    return {"pixel_values": pixel_values, "labels": labels}

def model_init(classes, class_to_idx):
    model = ViTForImageClassification.from_pretrained(
        pretrained_model_name_or_path="google/vit-base-patch16-224-in21k",
        num_labels=len(classes),
        id2label={idx: label for label, idx in class_to_idx.items()},
        ignore_mismatched_sizes=True
    )
    return model

def train():
    subset_train_dataset = torch.load('/app/subset_train_dataset.pt')
    subset_test_dataset = torch.load('/app/subset_test_dataset.pt')

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_processor.size['height'], image_processor.size['width'])),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])

    train_dataloader = DataLoader(
        subset_train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda x: collator(x, transform),
        drop_last=True
    )

    valid_dataloader = DataLoader(
        subset_test_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda x: collator(x, transform),
        drop_last=True
    )

    classes = subset_train_dataset.dataset.classes
    class_to_idx = subset_train_dataset.dataset.class_to_idx

    training_args = TrainingArguments(
        output_dir="/app/models/ViT-FashionMNIST",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.001,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="/app/logs",
        logging_steps=125,
        remove_unused_columns=False,
        seed=7
    )

    def compute_metrics(eval_pred):
        import evaluate
        import numpy as np
        metric = evaluate.load('f1')
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        macro_f1 = metric.compute(predictions=predictions, references=labels, average="macro")
        return macro_f1

    trainer = Trainer(
        model_init=lambda: model_init(classes, class_to_idx),
        args=training_args,
        train_dataset=subset_train_dataset,
        eval_dataset=subset_test_dataset,
        data_collator=lambda x: collator(x, transform),
        compute_metrics=compute_metrics,
        tokenizer=image_processor,
    )

    trainer.train()
    trainer.save_model('/app/models/vit_model.pth')

if __name__ == "__main__":
    train()
