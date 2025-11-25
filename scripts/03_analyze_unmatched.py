import sys
import pandas as pd
from pathlib import Path
from collections import Counter
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import Preprocessor

INPUT_FILE = Path("data/processed/pseudo_labels_v1.parquet")
OUTPUT_FILE = Path("outputs/unmatched_keywords_top200.txt")

def main():
    if not INPUT_FILE.exists():
        print(f"Файл не найден: {INPUT_FILE}")
        return

    print("Загружаем файл с псевдоразметкой...")
    df = pd.read_parquet(INPUT_FILE)

    if "category" not in df.columns:
        print("ОШИБКА: Нет колонки category!")
        return
    
    if "clean_text" in df.columns:
        text_col = "clean_text"
    else:
        text_col = df.columns[0]   # fallback

    print("Фильтруем неразмеченные строки...")
    unmatched = df[
        (df["category"].isna()) |
        (df["category"] == "") |
        (df["category"] == "Прочее / Требует доработки")
    ]

    print(f"Неразмеченных строк: {len(unmatched):,}")

    prep = Preprocessor()
    counter = Counter()

    print("Считаем частоты слов...")
    for text in unmatched[text_col].dropna():
        clean = prep.clean(text)
        words = clean.split()
        counter.update(words)

    top200 = counter.most_common(400)

    OUTPUT_FILE.parent.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for word, count in top200:
            f.write(f"{word}\t{count}\n")


if __name__ == "__main__":
    main()
