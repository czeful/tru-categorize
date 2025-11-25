import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TORCH_LOAD_REQUIRE_VERSION"] = "0"          
os.environ["HF_HUB_DISABLE_TORCH_LOAD"] = "1"           
os.environ["HF_HUB_ENABLE_HFTRANSFER"] = "1"             

BASE = Path(__file__).resolve().parent.parent
TRAIN_FILE = BASE / "data" / "processed" / "bert_train.parquet"
VAL_FILE = BASE / "data" / "processed" / "bert_val.parquet"
MODEL_DIR = BASE / "models" / "bert_classifier"

MODEL_NAME = "ai-forever/ruBert-base"     
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5


# DATASET
class BertDataset(Dataset):
    def __init__(self, df, tokenizer, label2id):
        self.texts = df["clean"].tolist()
        self.labels = df["category"].map(label2id).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        enc["labels"] = self.labels[idx]
        return enc


def main():
    print("\n=== TRAIN BERT CLASSIFIER ===")

    if not TRAIN_FILE.exists():
        raise FileNotFoundError("Нет train.parquet — запусти 08_prepare_bert_dataset.py")

    df_train = pd.read_parquet(TRAIN_FILE)
    df_val = pd.read_parquet(VAL_FILE)

    print(f"[DATA] Train: {len(df_train):,}, Val: {len(df_val):,}")
    print(f"[DATA] Классов: {df_train['category'].nunique()}")

    # Метки → индексы
    labels = sorted(df_train["category"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    print("[LABELS] Пример:", list(label2id.items())[:10])

    print(f"[MODEL] Загружаем tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    if hasattr(tokenizer, "_files_to_download"):
        tokenizer._files_to_download = {
            k: v for k, v in tokenizer._files_to_download.items()
            if v.endswith(".json") or v.endswith(".txt")
        }

    print(f"[MODEL] Загружаем BERT c safetensors: {MODEL_NAME}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        use_safetensors=True,                
        trust_remote_code=False,
        ignore_mismatched_sizes=True,
    )


    train_dataset = BertDataset(df_train, tokenizer, label2id)
    val_dataset = BertDataset(df_val, tokenizer, label2id)
    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
    output_dir=str(MODEL_DIR),
    eval_strategy="epoch",       
    save_strategy="epoch",             
    save_total_limit=2,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    metric_for_best_model="eval_loss",   
    greater_is_better=False,             
    load_best_model_at_end=True,         
    logging_steps=50,
    fp16=torch.cuda.is_available(),
)

    print(f"[GPU] GPU доступен: {torch.cuda.is_available()}")

    class SaveBestCallback(EarlyStoppingCallback):
        def __init__(self):
            super().__init__(early_stopping_patience=2)
            self.best_acc = 0

        def on_evaluate(self, args, state, control, **kwargs):
            metrics = kwargs.get("metrics", {})
            acc = metrics.get("eval_accuracy")

            if acc is not None and acc > self.best_acc:
                self.best_acc = acc
                print(f"[BEST] Новая лучшая модель! accuracy={acc:.4f}")
                trainer.save_model(MODEL_DIR)
            return control


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda e: {
            "accuracy": (e.predictions.argmax(-1) == e.label_ids).mean()
        },
        callbacks=[SaveBestCallback()],
    )

    print("\n=== START TRAINING ===")
    trainer.train()

    print("\n=== SAVING MODEL ===")
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"[DONE] Модель сохранена в {MODEL_DIR}")


if __name__ == "__main__":
    main()
