"""Варианты SMOTE: стандартный, Borderline, ADASYN, SMOTE-Tomek, SMOTE-ENN."""

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN

from balancing.base import Balancer


class SMOTEBalancer(Balancer):
    name = "smote"

    def __init__(self, random_state: int = 42, k_neighbors: int = 5):
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def fit_resample(self, X, y):
        sampler = SMOTE(random_state=self.random_state, k_neighbors=self.k_neighbors)
        return sampler.fit_resample(X, y)


class BorderlineSMOTEBalancer(Balancer):
    name = "borderline_smote"

    def __init__(self, random_state: int = 42, k_neighbors: int = 5):
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def fit_resample(self, X, y):
        sampler = BorderlineSMOTE(random_state=self.random_state, k_neighbors=self.k_neighbors)
        return sampler.fit_resample(X, y)


class ADASYNBalancer(Balancer):
    name = "adasyn"

    def __init__(self, random_state: int = 42, n_neighbors: int = 5):
        self.random_state = random_state
        self.n_neighbors = n_neighbors

    def fit_resample(self, X, y):
        sampler = ADASYN(random_state=self.random_state, n_neighbors=self.n_neighbors)
        return sampler.fit_resample(X, y)


class SMOTETomekBalancer(Balancer):
    name = "smote_tomek"

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit_resample(self, X, y):
        sampler = SMOTETomek(random_state=self.random_state)
        return sampler.fit_resample(X, y)


class SMOTEENNBalancer(Balancer):
    name = "smote_enn"

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit_resample(self, X, y):
        sampler = SMOTEENN(random_state=self.random_state)
        return sampler.fit_resample(X, y)
