from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def fit_time_series_model(X, y, *, alpha: float = 1.0) -> Pipeline:
    """Fit a baseline pipeline. Caller must supply training-only rows and predeclared features."""
    if len(X) != len(y) or len(X) == 0:
        raise ValueError("X and y must have the same nonzero length")
    model = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                      ("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])
    return model.fit(X, y)


def fit_regime_clusterer(X, *, n_clusters: int = 4, random_state: int = 7):
    """Fit a baseline unsupervised regime clusterer on the supplied training window only.

    Do not fit on the full history before walk-forward evaluation: cluster centroids themselves
    can leak future distributional information. Cluster IDs have no intrinsic economic meaning.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    if n_clusters < 2: raise ValueError("n_clusters must be >= 2")
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         StandardScaler(),
                         KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)).fit(X)
