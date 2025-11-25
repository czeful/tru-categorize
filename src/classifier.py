# src/classifier.py (v4 — production ready)

from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
from functools import lru_cache
import joblib

from src.preprocessor import Preprocessor
from src.anchors import AnchorExamples
from src.rule_engine import RuleEngine

class TfidfSVMClassifier:

    def __init__(self, model_path: str | Path = None):
        if model_path is None:
            model_path = (
                Path(__file__).resolve().parent.parent
                / "models"
                / "tfidf_svm.joblib"
            )
        else:
            model_path = Path(model_path)

        self.model_path = model_path
        self.vectorizer = None
        self.clf = None

        if model_path.exists():
            try:
                payload = joblib.load(model_path)
                self.vectorizer = payload["vectorizer"]
                self.clf = payload["classifier"]
                print(f"[ML] Загружена модель TF-IDF + SVM: {model_path}")
            except Exception as e:
                print(f"[ML] Ошибка загрузки модели {model_path}: {e}")
        else:
            print(f"[ML] Модель отсутствует → ML отключен")

    def is_ready(self) -> bool:
        return self.vectorizer is not None and self.clf is not None

    def predict(self, clean_text: str) -> Optional[str]:
        if not self.is_ready():
            return None

        if not clean_text.strip():
            return None

        X = self.vectorizer.transform([clean_text])
        return self.clf.predict(X)[0]



class TRUClassifier:

    def __init__(
        self,
        rules_path: str = "config/rules_kz_2025_maxcoverage_v2.yaml",
        enable_ml: bool = False,
        return_explanation: bool = False,
        ml_model_path: str | Path = None,
    ):
        self.preprocessor = Preprocessor()
        self.anchors = AnchorExamples()
        self.rule_engine = RuleEngine(rules_path)

        # Explanation
        self.return_explanation = return_explanation

        # ML layer
        self.enable_ml = enable_ml
        self.ml = (
            TfidfSVMClassifier(ml_model_path)
            if enable_ml
            else None
        )

        if self.enable_ml:
            ready = self.ml.is_ready() if self.ml else False
            print(f"[Classifier] ML слой включен: {ready}")

    @lru_cache(maxsize=200_000)
    def _cached_predict(self, clean_text: str) -> Tuple[str, Dict[str, Any]]:
        explanation = {}

        anchor = self.anchors.match(clean_text)
        if anchor:
            explanation["reason"] = "anchor_match"
            explanation["rule"] = anchor
            return anchor, explanation

        rule = self.rule_engine.match(clean_text)
        if rule:
            explanation["reason"] = "rule_match"
            explanation["rule"] = rule
            return rule, explanation
        if (
            self.enable_ml
            and self.ml is not None
            and self.ml.is_ready()
        ):
            ml_pred = self.ml.predict(clean_text)
            if ml_pred:
                explanation["reason"] = "ml_model"
                explanation["rule"] = None
                return ml_pred, explanation

        explanation["reason"] = "fallback_other"
        explanation["rule"] = None
        return "Прочее / Требует доработки", explanation


    def predict(self, text: str) -> Any:
        if not isinstance(text, str) or not text.strip():
            return "Пустое наименование"

        clean = self.preprocessor.clean(text)
        label, explanation = self._cached_predict(clean)

        if self.return_explanation:
            return {
                "input": text,
                "clean": clean,
                "label": label,
                "explanation": explanation,
            }

        return label
    
    def predict_batch(self, texts: List[str]) -> List[Any]:
        clean_list = self.preprocessor.clean_batch(texts)
        results = []

        for raw, clean in zip(texts, clean_list):

            if not clean:
                results.append("Пустое наименование")
                continue

            label, explanation = self._cached_predict(clean)

            if self.return_explanation:
                results.append({
                    "input": raw,
                    "clean": clean,
                    "label": label,
                    "explanation": explanation
                })
            else:
                results.append(label)

        return results


classifier = TRUClassifier()

