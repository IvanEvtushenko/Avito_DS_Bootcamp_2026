"""Токенизатор русского текста: (razdel) → фильтры → (pymorphy3) → стоп-слова.

Адаптация токенизатора из BM25-проекта автора: та же схема (нижний регистр →
токены → фильтр пунктуации/длины → лемма → стоп-слова → кэш лемм), но с
подключаемыми бэкендами, чтобы код работал и там, где `razdel`/`pymorphy3` не
установлены (план §5.1.5 — отключаемость):

- лемматизатор: `pymorphy3` (если импортируется) либо тождество (без лемм);
- токенизатор: `razdel` (если импортируется) либо регулярка по словам.

Один и тот же экземпляр используется и для статей, и для запросов — чтобы леммы
совпадали. Плейсхолдеры (`<MONEY>`, `<DATE>`, ...) есть только в запросах;
их замена на слово/удаление настраивается конфигом (EDA: в статьях их нет).
"""
from __future__ import annotations

import re
import string
from typing import Callable, Mapping

from ..logging_utils import get_logger

logger = get_logger("preprocess.normalize")

_PUNCT = set(string.punctuation + "«»—–…„“”‘’№")
# Допустимый «сырой» токен: буквы/цифры/подчёркивание/дефис.
_TOKEN_RE = re.compile(r"^[\w\-]+$", re.UNICODE)
# Регуляр-бэкенд токенизации: последовательности букв/цифр с внутренними дефисами.
_WORD_RE = re.compile(r"[0-9a-zа-яё]+(?:-[0-9a-zа-яё]+)*", re.IGNORECASE)

# Русские стоп-слова (список из BM25-проекта автора).
RU_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех",
    "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть",
    "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего",
    "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем", "хорошо",
    "свою", "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя",
    "такой", "им", "более", "всегда", "конечно", "всю", "между",
    "это", "также", "которые", "который", "которая", "которых", "которой",
}


def _resolve_word_tokenizer(backend: str) -> tuple[str, Callable[[str], list[str]]]:
    """Вернуть (имя бэкенда, функция text -> сырые токены)."""
    want_razdel = backend in ("auto", "razdel")
    if want_razdel:
        try:
            from razdel import tokenize as razdel_tokenize

            def tok(text: str) -> list[str]:
                return [t.text for t in razdel_tokenize(text)]

            return "razdel", tok
        except ImportError:
            if backend == "razdel":
                raise ImportError("tokenizer=razdel, но пакет razdel не установлен")

    def tok_regex(text: str) -> list[str]:
        return _WORD_RE.findall(text)

    return "regex", tok_regex


def _resolve_lemmatizer(backend: str) -> tuple[str, Callable[[str], str]]:
    """Вернуть (имя бэкенда, функция token -> лемма). Тождество, если лемм нет."""
    want_pymorphy = backend in ("auto", "pymorphy3")
    if want_pymorphy:
        try:
            import pymorphy3
            import pymorphy3_dicts_ru

            morph = pymorphy3.MorphAnalyzer(path=pymorphy3_dicts_ru.get_path())

            def lemma(token: str) -> str:
                parses = morph.parse(token)
                return parses[0].normal_form if parses else token

            return "pymorphy3", lemma
        except ImportError:
            if backend == "pymorphy3":
                raise ImportError("lemmatizer=pymorphy3, но pymorphy3/словарь не установлены")

    return "none", (lambda token: token)


class Tokenizer:
    """Текст → список лемм (нижний регистр, без пунктуации/стоп-слов).

    Параметры
    ---------
    lemmatizer    : "auto" | "pymorphy3" | "none".
    tokenizer     : "auto" | "razdel" | "regex".
    min_len       : минимальная длина токена и леммы.
    use_stopwords : выбрасывать ли русские стоп-слова.
    placeholders  : {"<MONEY>": "деньги", "<ID>": ""} — замена/удаление плейсхолдеров.
    """

    def __init__(
        self,
        *,
        lemmatizer: str = "auto",
        tokenizer: str = "auto",
        min_len: int = 2,
        use_stopwords: bool = True,
        placeholders: Mapping[str, str] | None = None,
    ) -> None:
        self.min_len = min_len
        self.stopwords = set(RU_STOPWORDS) if use_stopwords else set()
        self.placeholders = dict(placeholders or {})
        self._tokenizer_name, self._word_tokenize = _resolve_word_tokenizer(tokenizer)
        self._lemmatizer_name, self._lemmatize_raw = _resolve_lemmatizer(lemmatizer)
        self._cache: dict[str, str] = {}
        logger.info(
            "Tokenizer: tokenizer=%s lemmatizer=%s min_len=%d stopwords=%d placeholders=%d",
            self._tokenizer_name,
            self._lemmatizer_name,
            self.min_len,
            len(self.stopwords),
            len(self.placeholders),
        )

    def backend_info(self) -> dict[str, object]:
        """Активные бэкенды — для manifest.json и логов (наблюдаемость)."""
        return {
            "tokenizer": self._tokenizer_name,
            "lemmatizer": self._lemmatizer_name,
            "min_len": self.min_len,
            "use_stopwords": bool(self.stopwords),
            "placeholders": dict(self.placeholders),
        }

    def _apply_placeholders(self, text: str) -> str:
        # До приведения к нижнему регистру: ключи вида "<MONEY>" — заглавные.
        for placeholder, replacement in self.placeholders.items():
            if placeholder in text:
                text = text.replace(placeholder, f" {replacement} " if replacement else " ")
        return text

    def _lemma(self, token: str) -> str:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        lemma = self._lemmatize_raw(token)
        self._cache[token] = lemma
        return lemma

    def __call__(self, text: str) -> list[str]:
        text = self._apply_placeholders(text or "")
        out: list[str] = []
        for raw_token in self._word_tokenize(text):
            raw = raw_token.lower()
            if not raw or raw in _PUNCT or not _TOKEN_RE.match(raw) or len(raw) < self.min_len:
                continue
            lemma = self._lemma(raw)
            if lemma in self.stopwords or len(lemma) < self.min_len:
                continue
            out.append(lemma)
        return out
