import re
import string
import numpy as np

import nltk
for _r in ("stopwords", "wordnet", "punkt", "punkt_tab", "omw-1.4"):
    try:
        nltk.download(_r, quiet=True)
    except Exception:
        pass

try:
    from nltk.corpus import stopwords as _sw
    _NLTK_STOPWORDS = set(_sw.words("english"))
except Exception:
    _NLTK_STOPWORDS = {
        "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
        "yourself","yourselves","he","him","his","himself","she","her","hers",
        "herself","it","its","itself","they","them","their","theirs","themselves",
        "what","which","who","whom","this","that","these","those","am","is","are",
        "was","were","be","been","being","have","has","had","having","do","does",
        "did","doing","a","an","the","and","but","if","or","because","as","until",
        "while","of","at","by","for","with","about","against","between","into",
        "through","during","before","after","above","below","to","from","up","down",
        "in","out","on","off","over","under","again","further","then","once","here",
        "there","when","where","why","how","all","both","each","few","more","most",
        "other","some","such","own","same","than","too","very","s","t","can","will",
        "just","should","now","d","ll","m","o","re","ve","y","ain","ma",
    }

try:
    from nltk.stem import WordNetLemmatizer as _WNL
    _lemmatizer_instance = _WNL()
    _lemmatizer_instance.lemmatize("test")
    _WORDNET_AVAILABLE = True
except Exception:
    _WORDNET_AVAILABLE = False

try:
    from nltk.tokenize import word_tokenize as _nltk_word_tokenize
    _nltk_word_tokenize("test")
    _PUNKT_AVAILABLE = True
except Exception:
    _PUNKT_AVAILABLE = False


CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
    "'t": " not", "'ve": " have", "'m": " am",
    "it's": "it is", "i'm": "i am", "i've": "i have",
    "i'll": "i will", "i'd": "i would", "you're": "you are",
    "you've": "you have", "you'll": "you will", "you'd": "you would",
    "he's": "he is", "she's": "she is", "they're": "they are",
    "we're": "we are", "we've": "we have", "we'll": "we will",
    "that's": "that is", "what's": "what is", "who's": "who is",
    "there's": "there is", "let's": "let us", "don't": "do not",
    "doesn't": "does not", "didn't": "did not", "wasn't": "was not",
    "weren't": "were not", "isn't": "is not", "aren't": "are not",
    "haven't": "have not", "hasn't": "has not", "wouldn't": "would not",
    "couldn't": "could not", "shouldn't": "should not",
}


class TextPreprocessor:
    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize        = lemmatize and _WORDNET_AVAILABLE

        self._stop_words     = set(_NLTK_STOPWORDS)
        self._negation_words = {"no", "not", "nor", "neither", "never", "none"}
        self._stop_words    -= self._negation_words

        self._lemmatizer = _lemmatizer_instance if _WORDNET_AVAILABLE else None

        if not _PUNKT_AVAILABLE:
            print("[Preprocessor] NLTK punkt unavailable — using regex tokenizer.")
        if not _WORDNET_AVAILABLE:
            print("[Preprocessor] NLTK WordNet unavailable — lemmatization disabled.")

    def transform(self, texts):
        return [self._clean(t) for t in texts]

    def _clean(self, text: str) -> str:
        text = text.lower()
        text = self._remove_html(text)
        text = self._expand_contractions(text)
        text = self._remove_special_chars(text)
        tokens = self._tokenize(text)
        if self.remove_stopwords:
            tokens = self._filter_stopwords(tokens)
        if self.lemmatize:
            tokens = self._lemmatize_tokens(tokens)
        return " ".join(tokens)

    @staticmethod
    def _remove_html(text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text)

    @staticmethod
    def _expand_contractions(text: str) -> str:
        pattern = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS) + r")\b",
            flags=re.IGNORECASE,
        )
        return pattern.sub(lambda m: CONTRACTIONS.get(m.group(0).lower(), m.group(0)), text)

    @staticmethod
    def _remove_special_chars(text: str) -> str:
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _tokenize(text: str):
        if _PUNKT_AVAILABLE:
            return _nltk_word_tokenize(text)
        return re.findall(r"\b[a-z0-9]+\b", text)

    def _filter_stopwords(self, tokens):
        return [t for t in tokens if t not in self._stop_words and len(t) > 1]

    def _lemmatize_tokens(self, tokens):
        return [self._lemmatizer.lemmatize(t) for t in tokens]