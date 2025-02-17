import pickle
import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm
import torch

# dataset class to get the accepted format
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# load data
with open("data/train_dataset.pkl", "rb") as f:
    train_dataset = pickle.load(f)
with open("data/val_dataset.pkl", "rb") as f:
    val_dataset = pickle.load(f)
with open("data/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# datasets into dataframes
train_df = pd.DataFrame({
    "input_ids": train_dataset.encodings["input_ids"].numpy().tolist(),
    "attention_mask": train_dataset.encodings["attention_mask"].numpy().tolist(),
    "labels": train_dataset.labels,
})

val_df = pd.DataFrame({
    "input_ids": val_dataset.encodings["input_ids"].numpy().tolist(),
    "attention_mask": val_dataset.encodings["attention_mask"].numpy().tolist(),
    "labels": val_dataset.labels,
})

# combine train and val datasets
combined_df = pd.concat([train_df, val_df], ignore_index=True)

# balance dataset to 100.000 entries per party
def balance_to_100k(df):
    balanced_dfs = []
    for _, group in tqdm(df.groupby("labels"), desc="Balancing classes"):
        if len(group) > 100000:
            # undersample to 100.000 entries
            balanced_dfs.append(group.sample(n=100000, random_state=42))
        else:
            # oversample to 100.000 entries
            oversampled_group = group.sample(n=100000, replace=True, random_state=42)
            balanced_dfs.append(oversampled_group)
    balanced = pd.concat(balanced_dfs, ignore_index=True)
    print("Verteilung nach Balancierung:")
    print(balanced["labels"].value_counts())
    return balanced

balanced_df = balance_to_100k(combined_df)
print("Verteilung im balancierten Datensatz:")
print(balanced_df["labels"].value_counts())

# shuffle and split 80/20
balanced_df = shuffle(balanced_df, random_state=42)
train_fraction = 0.8
train_size = int(len(balanced_df) * train_fraction)
train_balanced_df = balanced_df.iloc[:train_size]
val_balanced_df = balanced_df.iloc[train_size:]

# check-ups
print("Verteilung im Trainingsdatensatz:")
print(train_balanced_df["labels"].value_counts())
print("Verteilung im Validierungsdatensatz:")
print(val_balanced_df["labels"].value_counts())

# to save datasets in batches
def save_in_batches(df, file_path, batch_size=1000):
    mode = 'w'  # start with overwriting
    header = True  # save header one time only
    total_saved = 0  # counter for saved entries
    for i in tqdm(range(0, len(df), batch_size), desc=f"Saving to {file_path}"):
        batch = df.iloc[i:i+batch_size]
        total_saved += len(batch)
        batch.to_csv(file_path, mode=mode, header=header, index=False)
        mode = 'a'  # attach data from the 2nd batch on
        header = False
    print(f"Gesamt gespeicherte Einträge in {file_path}: {total_saved}")

# save in batches
save_in_batches(train_balanced_df, "data/balanced_train_dataset.csv", batch_size=1000)
save_in_batches(val_balanced_df, "data/balanced_val_dataset.csv", batch_size=1000)

# convert back from df to dataset
train_balanced_dataset = {
    "input_ids": torch.tensor(train_balanced_df["input_ids"].tolist(), dtype=torch.long),
    "attention_mask": torch.tensor(train_balanced_df["attention_mask"].tolist(), dtype=torch.long),
    "labels": torch.tensor(train_balanced_df["labels"].tolist(), dtype=torch.long),
}
val_balanced_dataset = {
    "input_ids": torch.tensor(val_balanced_df["input_ids"].tolist(), dtype=torch.long),
    "attention_mask": torch.tensor(val_balanced_df["attention_mask"].tolist(), dtype=torch.long),
    "labels": torch.tensor(val_balanced_df["labels"].tolist(), dtype=torch.long),
}

# save file
with open("data/balanced_train_dataset.pkl", "wb") as f:
    pickle.dump(train_balanced_dataset, f)
    print("Balanced train dataset saved.")

with open("data/balanced_val_dataset.pkl", "wb") as f:
    pickle.dump(val_balanced_dataset, f)
    print("Balanced validation dataset saved.")

# size check-up
print(f"Original train dataset size: {len(train_dataset.labels)}")
print(f"Original validation dataset size: {len(val_dataset.labels)}")
print(f"Balanced train dataset size: {len(train_balanced_df)}")
print(f"Balanced validation dataset size: {len(val_balanced_df)}")
