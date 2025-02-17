import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from tqdm import tqdm

# load model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model = BertForSequenceClassification.from_pretrained("model-output/checkpoint-90000").to(device)

# load dataset
df = pd.read_csv('data/balanced_val_dataset.csv')

class CustomDataset(Dataset): #needed by bert for interpreting the data
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long).to(device),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long).to(device),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long).to(device),
        }

# from columns to lists
input_ids = df['input_ids'].apply(eval).tolist()
attention_mask = df['attention_mask'].apply(eval).tolist()
labels = df['labels'].tolist()

# create the dataset in the accepted format
eval_dataset = CustomDataset(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

# create DataLoader for batch processing
batch_size = 16
eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

# evaluate the model & get logits and predictions
logits_list = []
true_labels_list = []

model.eval()  #set model to evaluation mode
with torch.no_grad():
    for batch in tqdm(eval_loader, desc="Evaluating", unit="batch"):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        # forward pass
        logits = model(input_ids, attention_mask=attention_mask).logits
        logits_list.append(logits.cpu())  # Logits on CPU
        true_labels_list.extend(labels.cpu().numpy())

# combine all logits
all_logits = torch.cat(logits_list, dim=0)

# softmax over logits to calculate probabilities
probabilities = F.softmax(all_logits, dim=1).numpy()

# predicted classes
predicted_labels = torch.argmax(all_logits, dim=1).cpu().numpy()
true_labels = torch.tensor(true_labels_list)

# calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels, average='weighted')
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')

# calculate ROC AUC if possible
n_classes = len(set(true_labels.tolist()))
roc_auc = None
if n_classes > 1:
    true_labels_binarized = label_binarize(true_labels.numpy(), classes=range(n_classes))
    roc_auc = roc_auc_score(true_labels_binarized, probabilities, multi_class='ovr', average='weighted')

# plot ROC curve for top N classes
plt.figure()
for i in range(n_classes):
    fpr, tpr, thresholds = roc_curve(true_labels_binarized[:, i], probabilities[:, i])
    roc_auc_value = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (class {i}) (area = {roc_auc_value:.2f})')

# plot diagonal
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Multiclass')
plt.legend(loc='lower right')
plt.show()

# print the metrics
print("Accuracy:", accuracy)
print("F1-Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
if roc_auc is not None:
    print("ROC AUC:", roc_auc)
