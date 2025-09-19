import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


# --------------------
# Adapted dataset definition
# --------------------
class AdaptedConfDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Extract data – adapt to your data format
        thought_process = str(row["thought_process"]) if pd.notna(row["thought_process"]) else ""
        candidate = str(row["candidate"]) if pd.notna(row["candidate"]) else ""
        attribute_type = str(row["attribute_type"]) if "attribute_type" in row else ""

        # Build richer text input
        # Option 1: simple concatenation
        # text_input = f"Thought process: {thought_process} Candidate keyword: {candidate}"

        # Option 2: structured input (recommended)
        text_input = f"Attribute type: {attribute_type} Candidate keyword: {candidate}"
        context_input = thought_process

        # Use BERT sentence-pair input format
        inputs = self.tokenizer(
            text_input,
            context_input,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {key: val.squeeze(0) for key, val in inputs.items()}

        # Numerical features
        llm_conf = float(row["llm_conf"]) if pd.notna(row["llm_conf"]) else 0.0
        kb_conf = float(row["kb_conf"]) if pd.notna(row["kb_conf"]) else 0.0

        item["llm_conf"] = torch.tensor(llm_conf, dtype=torch.float)
        item["kb_conf"] = torch.tensor(kb_conf, dtype=torch.float)
        item["label"] = torch.tensor(float(row["label"]), dtype=torch.float)

        # Add attribute type info (optional, for later analysis)
        item["attribute_type"] = attribute_type

        return item


# --------------------
# Advanced fusion model definition
# --------------------
class AdvancedFusionModel(nn.Module):
    def __init__(self, bert_name="bert-base-chinese", fusion_type="multimodal_attention"):
        super(AdvancedFusionModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.hidden_size = self.bert.config.hidden_size
        self.fusion_type = fusion_type

        # Numeric feature processing network
        self.numeric_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.hidden_size)
        )

        if fusion_type == "simple_concat":
            self._build_simple_fusion()
        elif fusion_type == "multimodal_attention":
            self._build_multimodal_attention()
        elif fusion_type == "gated_fusion":
            self._build_gated_fusion()
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def _build_simple_fusion(self):
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def _build_multimodal_attention(self):
        self.text_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.numeric_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.attention_weights = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _build_gated_fusion(self):
        self.gate_network = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_size),
            nn.Sigmoid()
        )

        self.text_transform = nn.Linear(self.hidden_size, self.hidden_size)
        self.numeric_transform = nn.Linear(self.hidden_size, self.hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, token_type_ids, llm_conf, kb_conf):
        # Text feature extraction
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        text_features = bert_outputs.pooler_output

        # Numeric feature processing
        numeric_input = torch.stack([llm_conf, kb_conf], dim=1)
        numeric_features = self.numeric_encoder(numeric_input)

        if self.fusion_type == "simple_concat":
            fused = torch.cat([text_features, numeric_features], dim=1)
            return self.fusion(fused).squeeze(-1)
        elif self.fusion_type == "multimodal_attention":
            return self._forward_multimodal_attention(text_features, numeric_features)
        elif self.fusion_type == "gated_fusion":
            return self._forward_gated_fusion(text_features, numeric_features)

    def _forward_multimodal_attention(self, text_features, numeric_features):
        text_proj = self.text_proj(text_features)
        numeric_proj = self.numeric_proj(numeric_features)

        concat_features = torch.cat([text_proj, numeric_proj], dim=1)
        attention_weights = self.attention_weights(concat_features)

        weighted_text = text_proj * attention_weights[:, 0:1]
        weighted_numeric = numeric_proj * attention_weights[:, 1:2]
        fused_features = weighted_text + weighted_numeric

        return self.fusion(fused_features).squeeze(-1)

    def _forward_gated_fusion(self, text_features, numeric_features):
        concat_features = torch.cat([text_features, numeric_features], dim=1)
        gate = self.gate_network(concat_features)

        text_transformed = self.text_transform(text_features)
        numeric_transformed = self.numeric_transform(numeric_features)

        gated_text = gate * text_transformed
        gated_numeric = (1 - gate) * numeric_transformed
        fused_features = gated_text + gated_numeric

        return self.classifier(fused_features).squeeze(-1)


# --------------------
# Training and evaluation function
# --------------------
def train_and_eval_advanced(csv_path,
                            save_model_path="advanced_fusion_model.pt",
                            fusion_type="multimodal_attention",
                            epochs=20,
                            batch_size=16,
                            lr=2e-5,
                            bert_name="bert-base-chinese"):
    """
    Train and validate the advanced fusion model
    """
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded data: {len(df)} records")

    # Data quality check
    print("\nData quality check:")
    print(f"Missing thought_process: {df['thought_process'].isna().sum()}")
    print(f"Missing candidate: {df['candidate'].isna().sum()}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    # Data cleaning
    df = df.dropna(subset=['candidate', 'label'])
    df['thought_process'] = df['thought_process'].fillna("")
    df['llm_conf'] = df['llm_conf'].fillna(0.0)
    df['kb_conf'] = df['kb_conf'].fillna(0.0)

    print(f"Data after cleaning: {len(df)} records")

    tokenizer = BertTokenizer.from_pretrained(bert_name)

    # Create dataset
    dataset = AdaptedConfDataset(df, tokenizer)

    # Split into train/validation/test (8:1:1)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    print(f"Data split - Train: {train_size}, Validation: {val_size}, Test: {test_size}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AdvancedFusionModel(bert_name=bert_name, fusion_type=fusion_type).to(device)
    print(f"Fusion strategy: {fusion_type}")

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_auc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            llm_conf = batch["llm_conf"].to(device)
            kb_conf = batch["kb_conf"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, token_type_ids, llm_conf, kb_conf)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Validation
        model.eval()
        val_preds, val_gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                llm_conf = batch["llm_conf"].to(device)
                kb_conf = batch["kb_conf"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids, attention_mask, token_type_ids, llm_conf, kb_conf)
                val_preds.extend(outputs.cpu().numpy())
                val_gts.extend(labels.cpu().numpy())

        val_auc = roc_auc_score(val_gts, val_preds)
        val_acc = accuracy_score(val_gts, [1 if p > 0.5 else 0 for p in val_preds])

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_model_path)
            print(f"Best model saved (AUC: {best_auc:.4f})")

    # Test evaluation
    model.load_state_dict(torch.load(save_model_path, map_location=device))
    model.eval()
    test_preds, test_gts = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            llm_conf = batch["llm_conf"].to(device)
            kb_conf = batch["kb_conf"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids, llm_conf, kb_conf)
            test_preds.extend(outputs.cpu().numpy())
            test_gts.extend(labels.cpu().numpy())

    test_auc = roc_auc_score(test_gts, test_preds)
    test_acc = accuracy_score(test_gts, [1 if p > 0.5 else 0 for p in test_preds])

    print(f"\nFinal test results:")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    return model, test_auc, test_acc


# --------------------
# Predictor
# --------------------
class AdvancedConfidencePredictor:
    def __init__(self, model_path, fusion_type="multimodal_attention",
                 bert_name="bert-base-chinese", max_len=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.fusion_type = fusion_type

        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.model = AdvancedFusionModel(bert_name=bert_name, fusion_type=fusion_type).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        print(f"Advanced fusion model loaded (strategy: {fusion_type})")

    def predict_single(self, thought_process, candidate, attribute_type, llm_conf, kb_conf):
        """Predict for a single data instance"""
        text_input = f"Attribute type: {attribute_type} Candidate keyword: {candidate}"
        context_input = str(thought_process) if thought_process else ""

        inputs = self.tokenizer(
            text_input,
            context_input,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        token_type_ids = inputs["token_type_ids"].to(self.device)
        llm_conf_tensor = torch.tensor([float(llm_conf)], dtype=torch.float).to(self.device)
        kb_conf_tensor = torch.tensor([float(kb_conf)], dtype=torch.float).to(self.device)

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                llm_conf=llm_conf_tensor,
                kb_conf=kb_conf_tensor
            )

        return output.cpu().item()


# --------------------
# Example usage
# --------------------
def main():
    csv_path = './training_dataset.csv'  # dataset

    # Test different fusion strategies
    fusion_types = ["simple_concat", "multimodal_attention", "gated_fusion"]

    results = {}
    for fusion_type in fusion_types:
        print(f"\n{'=' * 50}")
        print(f"Training {fusion_type} model")
        print(f"{'=' * 50}")

        model_path = f"model/advanced_fusion_{fusion_type}.pt"

        try:
            model, test_auc, test_acc = train_and_eval_advanced(
                csv_path=csv_path,
                save_model_path=model_path,
                fusion_type=fusion_type,
                epochs=10,
                batch_size=16,
                lr=2e-5
            )

            results[fusion_type] = {'auc': test_auc, 'acc': test_acc}

        except Exception as e:
            print(f"Error training {fusion_type}: {e}")
            results[fusion_type] = {'error': str(e)}

    print(f"\n{'=' * 50}")
    print("Summary of all model results:")
    print(f"{'=' * 50}")
    for fusion_type, result in results.items():
        if 'error' in result:
            print(f"{fusion_type}: Training failed - {result['error']}")
        else:
            print(f"{fusion_type}: AUC={result['auc']:.4f}, Acc={result['acc']:.4f}")


if __name__ == "__main__":
    main()
