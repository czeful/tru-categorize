
import sys
import time
import chardet
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
import os
from itertools import combinations

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import Preprocessor

RAW_FILE_PATH = Path(
    r"C:\Users\zhayl\OneDrive - Astana IT University\Рабочий стол\Alibek\esf_fulll_202511211949.csv"
)
CHUNK_SIZE = 120_000
TOP_WORDS = 3000             
MIN_COUNT = 25               
MIN_NGRAM_COUNT = 80         
GENERATE_YAML = True         



def detect_encoding(path: Path):
    with open(path, "rb") as f:
        raw = f.read(100000)
        enc = chardet.detect(raw)["encoding"]
        print(f"[ENCODING] Определена кодировка: {enc}")
        return enc or "utf-8"


def extract_tokens(text: str):
    if not isinstance(text, str):
        return []

    clean = Preprocessor.clean(text)
    lem = Preprocessor.clean_and_lemmatize(clean)

    tokens = lem.split()
    return [t for t in tokens if len(t) > 2 and not t.isdigit()]


def build_ngrams(tokens, min_len=2, max_len=3):
    ngrams = []
    for n in range(min_len, max_len + 1):
        if len(tokens) >= n:
            for i in range(len(tokens) - n + 1):
                ng = " ".join(tokens[i:i+n])
                ngrams.append(ng)
    return ngrams


def find_text_column(df):
    candidates = ["наимен", "описание", "desc", "description", "тру", "name"]
    for col in df.columns:
        cl = col.lower()
        if any(key in cl for key in candidates):
            return col
    return df.columns[0]

def main():
    if not RAW_FILE_PATH.exists():
        print("[ERROR] Файл не найден:", RAW_FILE_PATH)
        return

    encoding = detect_encoding(RAW_FILE_PATH)
    for sep in [";", ",", "\t"]:
        try:
            print(f"[TRY] Разделитель '{sep}'...")
            df_iter = pd.read_csv(
                RAW_FILE_PATH,
                sep=sep,
                encoding=encoding,
                chunksize=CHUNK_SIZE,
                dtype=str,
                on_bad_lines="skip",
                engine="python",
                quoting=3
            )
            first_chunk = next(df_iter)
            print(f" Разделитель '{sep}'. Колонок: {len(first_chunk.columns)}")
            break
        except Exception as e:
            print(f"[FAILED] '{sep}': {e}")
            df_iter = None

    if df_iter is None:
        print("[ERROR] Не удалось определить разделитель")
        return

    text_col = find_text_column(first_chunk)
    print(f"[INFO] Найдена колонка текста: {text_col}")

    # Перезапускаем итератор уже подтверждённым sep
    df_iter = pd.read_csv(
        RAW_FILE_PATH,
        sep=sep,
        encoding=encoding,
        chunksize=CHUNK_SIZE,
        dtype=str,
        on_bad_lines="skip",
        engine="python",
        quoting=3
    )

    print("\n[START] Начинаем сканирование датасета...")
    t0 = time.time()

    token_counter = Counter()
    ngram_counter = Counter()

    chunk_index = 0

    for chunk in df_iter:
        chunk_index += 1
        print(f"\n[CHUNK] №{chunk_index:,} → {len(chunk):,} строк")

        texts = chunk[text_col].dropna().astype(str).tolist()

        # batch-clean + batch-lemmatize
        clean_batch = Preprocessor.clean_batch(texts)
        lemma_batch = Preprocessor.lemmatize_batch(clean_batch)

        for lem in lemma_batch:
            tokens = lem.split()

            token_counter.update(tokens)

            # биграммы + триграммы
            ngrams = build_ngrams(tokens)
            ngram_counter.update(ngrams)

    print(f"\n[OK] Обработка завершена за {time.time() - t0:.1f} сек")
    print(f"Всего уникальных слов: {len(token_counter):,}")
    print(f"Всего уникальных n-грамм: {len(ngram_counter):,}")


    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    top_words_path = out_dir / "top_keywords.txt"

    with open(top_words_path, "w", encoding="utf-8") as f:
        for tok, count in token_counter.most_common(TOP_WORDS):
            if count >= MIN_COUNT:
                f.write(f"{tok}: {count}\n")

    print(f"[OK] Сохранено top слова → {top_words_path}")

    if GENERATE_YAML:
        yaml_path = out_dir / "generated_rules.yaml"

        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("# Автоматически сгенерировано\n")
            f.write("# Используй как основу для ручной правки\n\n")

            for ng, cnt in ngram_counter.most_common(500):
                if cnt < MIN_NGRAM_COUNT:
                    continue

                tokens = ng.split()

                f.write("- pattern: \"{}\"\n".format(".*".join(tokens)))
                f.write("  category: \"???\"\n")
                f.write("  priority: 10\n")
                f.write("  apply_if: [{}]\n".format(
                    ", ".join([f"\"{t}\"" for t in tokens[:1]])
                ))
                f.write("  exclude_if: []\n\n")




if __name__ == "__main__":
    main()
