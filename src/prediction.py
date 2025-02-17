import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import json
import sys

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_checkpoint():
    #checkpoint_path = "model-output/checkpoint-90000"
    #print(f"Debug: Versuche Modell aus {checkpoint_path} zu laden...")
    model = BertForSequenceClassification.from_pretrained("julietteyek/bundestag-speeches-lite")
    print(f"Debug: Modell erfolgreich geladen.")
    model.to(device)
    model.eval()

    label_encoder = {
        0: "AfD",
        1: "BÜNDNIS 90/DIE GRÜNEN",
        2: "CDU/CSU",
        3: "DIE LINKE",
        4: "FDP",
        5: "SPD"
    }
    return model, label_encoder

def predict_party(model, input_text, label_encoder):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).cpu().detach().numpy()[0]

    party_probs = {label_encoder[idx]: float(prob) for idx, prob in enumerate(probs)}
    return dict(sorted(party_probs.items(), key=lambda item: item[1], reverse=True))

if __name__ == "__main__":
    model, label_encoder = load_checkpoint()

    input_text = sys.stdin.read().strip()
    if input_text:
        try:
            predictions = predict_party(model, input_text, label_encoder)
            print(json.dumps(predictions))
        except Exception as e:
            print(json.dumps({"error": str(e)}))
    else:
        print(json.dumps({"error": "No input text provided"}))
