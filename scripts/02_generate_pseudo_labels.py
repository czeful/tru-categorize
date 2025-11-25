# scripts/02_generate_pseudo_labels.py  (v3 — стабильная версия)

import pandas as pd
from pathlib import Path
import sys
import os
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import Preprocessor
from src.classifier import TRUClassifier

tqdm.pandas()

RAW_FILE = Path(
    r"C:\Users\zhayl\OneDrive - Astana IT University\Рабочий стол\Alibek\esf_fulll_202511211949.csv"
)
TEXT_COLUMN = "DESCRIPTION"
OUTPUT_FILE = Path("data/processed/pseudo_labels_v1.parquet")
BATCH_SIZE = 10_000


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Не найден файл: {RAW_FILE}")

    print("Загружаем классификатор (anchors + rules)...")
    classifier = TRUClassifier(enable_ml=False)

    print(f"Читаем {RAW_FILE.name} по частям...")
    chunks = pd.read_csv(RAW_FILE, usecols=[TEXT_COLUMN], chunksize=BATCH_SIZE)

    results = []
    total_rows_read = 0

    for chunk in tqdm(chunks, desc="Псевдоразметка"):
        texts = chunk[TEXT_COLUMN].astype(str).fillna("").tolist()
        preds = classifier.predict_batch(texts)
        total_rows_read += len(texts)

        results.extend(
            {
                "text": text,
                "category": pred,
                "source": "rules"
            }
            for text, pred in zip(texts, preds)
        )

    print("\nОбработка и дедупликация...")
    df = pd.DataFrame(results)
    df["category"] = df["category"].replace(
        {
            None: "Прочее / Требует доработки",
            "": "Прочее / Требует доработки",
            "Пустое наименование": "Прочее / Требует доработки",
        }
    )
    df["is_labeled"] = df["category"] != "Прочее / Требует доработки"

    df.sort_values(by="is_labeled", ascending=False, inplace=True)
    df.drop_duplicates(subset=["text"], keep="first", inplace=True)

    df.drop(columns=["is_labeled"], inplace=True)

    # Сохранение
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    total = len(df)
    labeled = (df["category"] != "Прочее / Требует доработки").sum()
    coverage = labeled / total * 100

    print("\n--- РЕЗУЛЬТАТЫ ПСЕВДОРАЗМЕТКИ ---")
    print(f"Всего уникальных строк: {total:,}")
    print(f"Размечено правилом: {labeled:,}")
    print(f"Покрытие: {coverage:.1f}%")
    print(f"НЕ распознано: {total - labeled:,}")
    print(f"Файл сохранён: {OUTPUT_FILE}")
    print(f"Уникальных категорий: {df['category'].nunique()}")

    # Топ 20 категорий
    print("\nТоп-20 категорий:")
    print(df["category"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
