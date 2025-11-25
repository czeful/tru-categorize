

import pandas as pd
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessor import Preprocessor
from src.classifier import TfidfSVMClassifier



BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "pseudo_labels_v1.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "pseudo_labels_v2.parquet"

TEXT_COL = "text"
LABEL_COL = "category"
SOURCE_COL = "source"

FALLBACK_LABEL = "Прочее / Требует доработки"


def main():
    print("=== GENERATE FULL LABELS (rules + ML) ===")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Входной файл не найден: {INPUT_PATH}")

    # 1. Загружаем v1
    print(f"[LOAD] Читаем {INPUT_PATH} ...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"[LOAD] Всего строк: {len(df):,}")

    required_cols = {TEXT_COL, LABEL_COL, SOURCE_COL}
    miss = required_cols.difference(df.columns)
    if miss:
        raise ValueError(f"Не хватает колонок {miss} в {INPUT_PATH}. Есть: {list(df.columns)}")

    total = len(df)
    n_fallback_before = (df[LABEL_COL] == FALLBACK_LABEL).sum()
    coverage_before = (total - n_fallback_before) / total * 100

    print(f"[STATS:BEFORE] 'Прочее / Требует доработки': {n_fallback_before:,}")
    print(f"[STATS:BEFORE] Покрытие (rules only):       {coverage_before:.2f}%")

    mask_fallback = df[LABEL_COL] == FALLBACK_LABEL
    df_ml = df[mask_fallback].copy()

    print(f"[ML] Кандидатов для ML: {len(df_ml):,}")

    if df_ml.empty:
        df.to_parquet(OUTPUT_PATH, index=False)
        print(f"[SAVE] {OUTPUT_PATH}")
        return

    
    df_ml["clean"] = df_ml[TEXT_COL].astype(str).apply(Preprocessor.clean)

    before_clean = len(df_ml)
    df_ml = df_ml[df_ml["clean"].str.len() > 0].copy()
    after_clean = len(df_ml)
    print(f"[PREP] После очистки осталось: {after_clean:,} (удалено пустых: {before_clean - after_clean:,})")

    if df_ml.empty:
        print("[ML] После очистки кандидатов нет → сохраняем оригинал")
        df.to_parquet(OUTPUT_PATH, index=False)
        print(f"[SAVE] {OUTPUT_PATH}")
        return

    print("[ML] Загружаем TF-IDF + SVM модель...")
    ml = TfidfSVMClassifier()

    if not ml.is_ready():
        raise RuntimeError(
            "ML модель не готова (vectorizer или classifier = None). "
            "Убедись, что ты запускал scripts/05_train_tfidf_svm.py и файл models/tfidf_svm.joblib существует."
        )

    print("[ML] Строим матрицу TF-IDF и предсказываем категории...")

    clean_texts = df_ml["clean"].tolist()

    # Векторизация
    X = ml.vectorizer.transform(clean_texts)

    # Предсказания
    ml_preds = ml.clf.predict(X)

    df_ml["ml_category"] = ml_preds

    df_out = df.copy()

    # Только для индексов, которые остались после очистки
    idx_to_update = df_ml.index

    df_out.loc[idx_to_update, LABEL_COL] = df_ml["ml_category"]
    df_out.loc[idx_to_update, SOURCE_COL] = "ml"

    n_fallback_after = (df_out[LABEL_COL] == FALLBACK_LABEL).sum()
    coverage_after = (total - n_fallback_after) / total * 100

    print("=== STATS AFTER ML ===")
    print(f"[STATS:AFTER] 'Прочее / Требует доработки': {n_fallback_after:,}")
    print(f"[STATS:AFTER] Покрытие (rules + ML):        {coverage_after:.2f}%")
    print(f"[DELTA] Улучшение покрытия:                 {coverage_after - coverage_before:.2f} п.п.")

    print("\n[STATS] Источники после ML (source):")
    print(df_out[SOURCE_COL].value_counts())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n[SAVE] Финальная разметка сохранена в: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
