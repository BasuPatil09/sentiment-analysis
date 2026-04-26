from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import FeatureUnion
import numpy as np


class FeatureExtractor:
    STRATEGIES = ["tfidf_word", "tfidf_char", "tfidf_combo", "bow"]

    def __init__(self, strategy: str = "tfidf_word", max_features: int = 50_000):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")
        self.strategy     = strategy
        self.max_features = max_features
        self.vectorizer   = self._build_vectorizer()

    def fit_transform(self, texts):
        print(f"[FeatureExtractor] Fitting '{self.strategy}' on {len(texts)} samples...")
        X = self.vectorizer.fit_transform(texts)
        print(f"[FeatureExtractor] Feature matrix shape: {X.shape}")
        return X

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

    def _build_vectorizer(self):
        if self.strategy == "tfidf_word":
            return TfidfVectorizer(
                analyzer      = "word",
                ngram_range   = (1, 2),
                max_features  = self.max_features,
                min_df        = 2,
                max_df        = 0.95,
                sublinear_tf  = True,
                strip_accents = "unicode",
            )

        elif self.strategy == "tfidf_char":
            return TfidfVectorizer(
                analyzer     = "char_wb",
                ngram_range  = (3, 5),
                max_features = self.max_features,
                min_df       = 3,
                sublinear_tf = True,
            )

        elif self.strategy == "tfidf_combo":
            word_vec = TfidfVectorizer(
                analyzer     = "word",
                ngram_range  = (1, 2),
                max_features = self.max_features // 2,
                min_df       = 2,
                max_df       = 0.95,
                sublinear_tf = True,
            )
            char_vec = TfidfVectorizer(
                analyzer     = "char_wb",
                ngram_range  = (3, 5),
                max_features = self.max_features // 2,
                min_df       = 3,
                sublinear_tf = True,
            )
            return FeatureUnion([("word", word_vec), ("char", char_vec)])

        elif self.strategy == "bow":
            return CountVectorizer(
                analyzer     = "word",
                ngram_range  = (1, 2),
                max_features = self.max_features,
                min_df       = 2,
                max_df       = 0.95,
            )