# src/anchors.py
import re
from typing import Optional


class AnchorExamples:
    ANCHORS = {
        "сопровождение.*сап|1с|функциональн": "Услуги → Сопровождение ИС",
        "принтер": "Оргтехника → Принтеры",
        "бумага.*офис": "Канцелярия → Бумага",
        "молоко|кефир|йогурт|нектар|буренкино|palma": "Продукты питания",
    }

    # Компилируем один раз при загрузке модуля
    _compiled = {
        re.compile(pattern, flags=re.IGNORECASE): category
        for pattern, category in ANCHORS.items()
    }

    @staticmethod
    def match(text: str) -> Optional[str]:
        for regex, category in AnchorExamples._compiled.items():
            if regex.search(text):
                return category
        return None

    @staticmethod
    def is_tz_example(text: str) -> bool:
        """Проверка, относится ли строка к примерам из ТЗ"""
        return AnchorExamples.match(text) is not None