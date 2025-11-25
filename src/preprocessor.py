# src/preprocessor.py
import re
import unicodedata
from functools import lru_cache
from typing import List


try:
    from natasha import (
        Segmenter,
        MorphVocab,
        NewsEmbedding,
        NewsMorphTagger,
        Doc,
    )
    emb = NewsEmbedding()
    segmenter = Segmenter()
    morph_tagger = NewsMorphTagger(emb)
    morph_vocab = MorphVocab()
    NATASHA_AVAILABLE = True
    print("[Preprocessor] Natasha загружена — максимальное качество лемматизации")

except ImportError:
    NATASHA_AVAILABLE = False
    print("[Preprocessor] Natasha не найдена → используется pymorphy3")

import pymorphy3
pymorphy = pymorphy3.MorphAnalyzer(lang='ru')


class Preprocessor:

    DASHES = re.compile(r"[‐-–—−]+")      
    SLASHES = re.compile(r"[⁄∕]")         


    ARTICUL_PATTERN = re.compile(
        r"^\s*(?:[A-Z0-9]{4,}|[0-9]{3,}|[A-Z]{2,}[0-9]{2,}|[0-9]{2,}[A-Z]{2,})\b[-_/]?\b",
    )

    DATE_TAIL_PATTERN = re.compile(
        r"\s+(за|от|по|на)\s+(период\s+)?(январь|февраль|март|апрель|май|июнь|"
        r"июль|август|сентябрь|октябрь|ноябрь|декабрь|\d{4}|\d{2}\.\d{2}\.\d{4}).*$",
        flags=re.IGNORECASE
    )

    ALLOWED_CHARS = re.compile(
        r"[^a-zA-Zа-яА-Я0-9\s\.\,\%\-\+\/\(\)]"
    )

    @staticmethod
    def clean_raw(text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = unicodedata.normalize("NFC", text)

        # Унификация дефисов и слэшей
        text = Preprocessor.DASHES.sub("-", text)
        text = Preprocessor.SLASHES.sub("/", text)

        return text.strip()

    @staticmethod
    def clean(text: str) -> str:
        text = Preprocessor.clean_raw(text)

        if not text:
            return ""

        text = Preprocessor.ARTICUL_PATTERN.sub(" ", text)
        text = Preprocessor.DATE_TAIL_PATTERN.sub("", text)
        text = Preprocessor.ALLOWED_CHARS.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text.lower()

    # Кэшируем лемматизацию 
    @staticmethod
    @lru_cache(maxsize=100_000)
    def _lemmatize_word(word: str) -> str:
        p = pymorphy.parse(word)[0]
        return p.normal_form

    @staticmethod
    def clean_and_lemmatize(text: str) -> str:
    
        text = Preprocessor.clean(text)
        if not text:
            return ""

        words = text.split()

    
        if NATASHA_AVAILABLE:
            try:
                doc = Doc(" ".join(words))
                doc.segment(segmenter)
                doc.tag_morph(morph_tagger)

                for token in doc.tokens:
                    token.lemmatize(morph_vocab)

                lemmas = [t.lemma for t in doc.tokens if t.lemma]
                return " ".join(lemmas)

            except Exception as e:
                print(f"[Natasha fallback] Ошибка: {e}")

        lemmas = [Preprocessor._lemmatize_word(w) for w in words]
        return " ".join(lemmas)

    @staticmethod
    def clean_batch(texts: List[str]) -> List[str]:
        return [Preprocessor.clean(t) for t in texts]

    @staticmethod
    def lemmatize_batch(texts: List[str]) -> List[str]:
        return [Preprocessor.clean_and_lemmatize(t) for t in texts]
