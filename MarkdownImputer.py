from sklearn.base import BaseEstimator, TransformerMixin


class MarkdownImputer(BaseEstimator, TransformerMixin):
    """
    Fills missing markdown/promotion columns with a constant (default 0).
    NaN in these columns typically means "no markdown was running".
    """

    def __init__(
        self,
        columns=("MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"),
        fill_value=0,
    ):
        self.columns = list(columns)
        self.fill_value = fill_value

    def fit(self, X, y=None):
        # No stats need to be learned, but sklearn's fitted-check
        # looks for a trailing-underscore attribute — this satisfies it.
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].fillna(self.fill_value)
        return X