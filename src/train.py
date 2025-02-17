import pickle
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load datasets
with open("data/balanced_train_dataset.pkl", "rb") as f:
    train_data = pickle.load(f)
with open("data/balanced_val_dataset.pkl", "rb") as f:
    val_data = pickle.load(f)
with open("data/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

num_labels = len(label_encoder.classes_)
print(f"Number of labels: {num_labels}")

# convert to accepted dataset format
train_dataset = Dataset.from_dict({
    "input_ids": train_data["input_ids"].tolist(),
    "attention_mask": train_data["attention_mask"].tolist(),
    "labels": train_data["labels"].tolist(),
})

val_dataset = Dataset.from_dict({
    "input_ids": val_data["input_ids"].tolist(),
    "attention_mask": val_data["attention_mask"].tolist(),
    "labels": val_data["labels"].tolist(),
})

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels).to(device)

# set Training arguments:
training_args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="no",  # Keine Evaluation während des Trainings
    save_steps=500,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir="logs",
    logging_steps=100,
    save_total_limit=10,
    load_best_model_at_end=False,  # Keine automatische Auswahl des besten Modells
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

if __name__ == "__main__":
    trainer.train()
    print("Training completed.")