
import re
import yaml
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any


class RuleEngine:
    def __init__(
        self,
        rules_path: str = "config/rules.yaml",
        explain_mode: bool = False,
        collect_stats: bool = True,
    ):
        self.rules_path = Path(rules_path)
        self.explain_mode = explain_mode
        self.collect_stats = collect_stats

  
        self.stats = {}

        self.compiled_rules = self._load_and_compile_rules()

    def _load_and_compile_rules(self) -> List[Dict[str, Any]]:
        if not self.rules_path.exists():
            raise FileNotFoundError(f"Файл правил не найден: {self.rules_path}")

        with open(self.rules_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not raw:
            return []

        compiled = []


        for item in raw:
            if isinstance(item, dict) and "pattern" in item:
                pattern = item["pattern"]
                category = item["category"]
                priority = item.get("priority", 0)
                apply_if = item.get("apply_if", [])
                exclude_if = item.get("exclude_if", [])
            elif isinstance(item, dict):

                (pattern, category), = item.items()
                priority = 0
                apply_if = []
                exclude_if = []
            else:
                print("[WARNING] Неизвестный формат правила:", item)
                continue

            try:
                regex = re.compile(pattern, flags=re.IGNORECASE)
            except re.error as e:
                print(f"[WARNING] Ошибка в паттерне '{pattern}': {e}")
                continue

            compiled.append({
                "pattern": pattern,
                "regex": regex,
                "category": category,
                "priority": priority,
                "apply_if": apply_if,
                "exclude_if": exclude_if,
            })

        # Сортировка по убыванию priority
        compiled.sort(key=lambda x: x["priority"], reverse=True)

        print(f"[RuleEngine] Загружено {len(compiled)} правил (приоритетная сортировка)")
        return compiled


    def match(self, text: str) -> Optional[str]:

        for rule in self.compiled_rules:

            if rule["apply_if"]:
                if not all(word in text for word in rule["apply_if"]):
                    continue

  
            if rule["exclude_if"]:
                if any(word in text for word in rule["exclude_if"]):
                    continue

            if rule["regex"].search(text):
                pattern = rule["pattern"]
                category = rule["category"]

                if self.collect_stats:
                    self.stats[pattern] = self.stats.get(pattern, 0) + 1

                if self.explain_mode:
                    return {
                        "category": category,
                        "pattern": pattern,
                        "priority": rule["priority"],
                        "apply_if": rule["apply_if"],
                        "exclude_if": rule["exclude_if"],
                    }

                return category

        return None

    def reload(self):
        self.compiled_rules = self._load_and_compile_rules()
        print(f"[RuleEngine] Правила перезагружены: {len(self.compiled_rules)} шт.")

    def get_stats(self) -> Dict[str, int]:

        return dict(sorted(self.stats.items(), key=lambda x: -x[1]))
