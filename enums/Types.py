from enum import Enum


class VectorizerType(Enum):
    HASHING = "HASHING_VECTORIZER"
    TFIDF = "TFIDF"
    COUNT = "COUNT_VECTORIZER"


class SemiSupervisedAlgorithms(Enum):
    LABEL_PROPAGATION = "PROPAGATION"
    LABEL_SPREADING = "SPREADING"
