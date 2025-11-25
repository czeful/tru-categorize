import pandas as pd
import random
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessor import Preprocessor
from src.rule_engine import RuleEngine
from src.classifier import TRUClassifier


SAMPLE_SIZE = 1000  # можно менять


def detect_triggered_rule(clean_text: str, rule_engine):

    best_rule = None
    best_priority = -10_000

    for rule in rule_engine.compiled_rules:
        regex = rule["regex"]
        category = rule["category"]
        priority = rule["priority"]
        apply_if = rule.get("apply_if", [])
        exclude_if = rule.get("exclude_if", [])

    
        if not regex.search(clean_text):
            continue
        if apply_if:
            if not all(token in clean_text for token in apply_if):
                continue

        if exclude_if:
            if any(token in clean_text for token in exclude_if):
                continue

        if priority > best_priority:
            best_priority = priority
            best_rule = rule

    if best_rule:
        return best_rule["pattern"], best_rule["category"]

    return None, None


def main():
    parquet_path = Path("data/processed/pseudo_labels_v1.parquet")

    if not parquet_path.exists():
        print("Файл pseudo_labels_v1.parquet не найден")
        return

    print("Загружаем файл...")
    df = pd.read_parquet(parquet_path)
    df_labeled = df[df["category"] != "Прочее / Требует доработки"]

    if len(df_labeled) < SAMPLE_SIZE:
        sample_size = len(df_labeled)
    else:
        sample_size = SAMPLE_SIZE

    print(f"Всего размеченных строк: {len(df_labeled):,}")
    print(f"Берём случайную выборку: {sample_size}")

    sample = df_labeled.sample(sample_size, random_state=42)


    classifier = TRUClassifier()
    pre = classifier.preprocessor
    rule_engine = classifier.rule_engine

    print("Определяем, какие правила сработали...")

    triggered_patterns = []
    triggered_categories = []

    for text in sample["text"]:
        clean = pre.clean(text)
        rule_pattern, rule_category = detect_triggered_rule(clean, rule_engine)

        triggered_patterns.append(rule_pattern)
        triggered_categories.append(rule_category)

    sample["clean_text"] = sample["text"].apply(pre.clean)
    sample["triggered_pattern"] = triggered_patterns
    sample["triggered_rule_category"] = triggered_categories

    output_path = Path("outputs/manual_check_sample.xlsx")
    output_path.parent.mkdir(exist_ok=True)

    sample.to_excel(output_path, index=False)



if __name__ == "__main__":
    main()
