

import pandas as pd
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessor import Preprocessor



BASE = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE / "data" / "processed" / "pseudo_labels_v2.parquet"
OUTPUT_DIR = BASE / "data" / "processed"

MIN_SAMPLES_PER_CLASS = 200       
MAX_SAMPLES_PER_CLASS = 50_000    
VAL_SIZE = 0.1                    

TEXT_COL = "text"
LABEL_COL = "category"

def main():
    print("\n=== PREPARE BERT DATASET ===")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {INPUT_PATH}")

    print(f"[LOAD] Загружаем {INPUT_PATH} ...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"[LOAD] Всего строк: {len(df):,}")


    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df = df[df[LABEL_COL].str.startswith("МУСОР") == False]

    print(f"[CLEAN] После удаления мусора: {len(df):,}")

    print("[PREP] Чистим текст...")
    df["clean"] = df[TEXT_COL].astype(str).apply(Preprocessor.clean)
    before = len(df)
    df = df[df["clean"].str.strip().ne("")]
    print(f"[PREP] После очистки удалено: {before - len(df):,}")

    vc = df[LABEL_COL].value_counts()
    good_labels = vc[vc >= MIN_SAMPLES_PER_CLASS].index
    df = df[df[LABEL_COL].isin(good_labels)]

    print(f"[FILTER] После фильтра классов (>= {MIN_SAMPLES_PER_CLASS}): {len(df):,}")
    print(f"[FILTER] Количество классов: {df[LABEL_COL].nunique()}")

    print("[BALANCE] Применяем max-per-class...")
    parts = []
    for label, group in df.groupby(LABEL_COL):
        if len(group) > MAX_SAMPLES_PER_CLASS:
            parts.append(group.sample(MAX_SAMPLES_PER_CLASS, random_state=42))
        else:
            parts.append(group)
    df = pd.concat(parts).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"[BALANCE] После балансировки: {len(df):,}")

    print("[SPLIT] Создаём train/val (stratified)...")
    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=42,
        stratify=df[LABEL_COL]
    )

    print(f"[SPLIT] Train: {len(train_df):,}")
    print(f"[SPLIT] Val:   {len(val_df):,}")


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "bert_train.parquet"
    val_path = OUTPUT_DIR / "bert_val.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"[SAVE] Train → {train_path}")
    print(f"[SAVE] Val   → {val_path}")

    print("\n=== DONE: BERT DATASET READY ===")


if __name__ == "__main__":
    main()
