
import os
from pathlib import Path
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessor import Preprocessor


BASE_DIR = Path(__file__).resolve().parent.parent

PSEUDO_LABELS_PATH = BASE_DIR / "data" / "processed" / "pseudo_labels_v1.parquet"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "tfidf_svm.joblib"

TEXT_COL = "text"       
LABEL_COL = "category" 

MIN_SAMPLES_PER_CLASS = 50

MAX_SAMPLES_PER_CLASS = 100_000


def load_data() -> pd.DataFrame:
    if not PSEUDO_LABELS_PATH.exists():
        raise FileNotFoundError(f"Файл с псевдоработкой не найден: {PSEUDO_LABELS_PATH}")

    print(f"[DATA] Загружаем псевдоразметку из {PSEUDO_LABELS_PATH} ...")
    df = pd.read_parquet(PSEUDO_LABELS_PATH)

    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"Ожидаю'{TEXT_COL}' и '{LABEL_COL}' в {PSEUDO_LABELS_PATH}, "
                         f"но есть: {list(df.columns)}")

    # Удаляем пустые
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])

    # Удаляем мусорные категории
    mask_not_garbage = ~df[LABEL_COL].str.startswith("МУСОР")
    df = df[mask_not_garbage]

    print(f"[DATA] После фильтра мусора: {len(df):,} строк")

    # Фильтрация редко встречающихся классов
    vc = df[LABEL_COL].value_counts()
    keep_cats = vc[vc >= MIN_SAMPLES_PER_CLASS].index
    df = df[df[LABEL_COL].isin(keep_cats)]

    print(f"[DATA] После фильтра редких классов (<{MIN_SAMPLES_PER_CLASS}): {len(df):,} строк")
    print(f"[DATA] Кол-во классов: {df[LABEL_COL].nunique()}")

    # Балансировка по максимуму
    balanced = []
    for cat, group in df.groupby(LABEL_COL):
        if len(group) > MAX_SAMPLES_PER_CLASS:
            balanced.append(group.sample(MAX_SAMPLES_PER_CLASS, random_state=42))
        else:
            balanced.append(group)
    df_balanced = pd.concat(balanced, ignore_index=True)

    print(f"[DATA] После балансировки (max {MAX_SAMPLES_PER_CLASS}/класс): {len(df_balanced):,} строк")

    # Перемешиваем
    df_balanced = df_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

    return df_balanced


def preprocess_texts(df: pd.DataFrame) -> pd.Series:
    print("[PREP] Применяем Preprocessor.clean к текстам...")
    df["clean_text"] = df[TEXT_COL].astype(str).apply(Preprocessor.clean)
    before = len(df)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    after = len(df)
    print(f"[PREP] После очистки удалено пустых строк: {before - after}")
    return df


def train_model(df: pd.DataFrame):
    X_text = df["clean_text"].tolist()
    y = df[LABEL_COL].tolist()

    print("[SPLIT] train / val = 90 / 10 (stratified)")
    X_train, X_val, y_train, y_val = train_test_split(
        X_text,
        y,
        test_size=0.1,
        random_state=42,
        stratify=y
    )

    print(f"[SPLIT] train: {len(X_train):,}, val: {len(X_val):,}")

    print("[VEC] Обучаем TfidfVectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
        max_features=200_000,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    print(f"[VEC] Размер словаря: {len(vectorizer.vocabulary_):,}")

    clf = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=10_000,
    )

    clf.fit(X_train_tfidf, y_train)

    print("[EVAL] Оцениваем на валидации...")
    y_pred = clf.predict(X_val_tfidf)

    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro")
    f1_weighted = f1_score(y_val, y_pred, average="weighted")

    print(f"[EVAL] Accuracy:      {acc:.4f}")
    print(f"[EVAL] F1 (macro):    {f1_macro:.4f}")
    print(f"[EVAL] F1 (weighted): {f1_weighted:.4f}")
    print()
    print("[EVAL] TOP-репорт по классам:")
    print(classification_report(y_val, y_pred, digits=3))

    return vectorizer, clf


def save_model(vectorizer, clf):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "vectorizer": vectorizer,
        "classifier": clf,
    }

    joblib.dump(payload, MODEL_PATH)
    print(f"[SAVE] Модель и vectorizer сохранены в {MODEL_PATH}")


def main():
    df = load_data()
    df = preprocess_texts(df)
    vectorizer, clf = train_model(df)
    save_model(vectorizer, clf)


if __name__ == "__main__":
    main()
