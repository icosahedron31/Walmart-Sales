from sklearn.base import BaseEstimator, TransformerMixin


class EconomicIndicatorImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=("CPI", "Unemployment")):
        self.columns = list(columns)

    def fit(self, X, y=None):
        X = X.copy()
        X = X.sort_values(["Store", "Date"])

        self.last_values_ = (
            X.groupby("Store")[self.columns]
             .last()
        )

        self.global_values_ = X[self.columns].mean()

        return self

    def transform(self, X):
        X = X.copy()
        X = X.sort_values(["Store", "Date"])

        for col in self.columns:
            # Fill gaps within each store
            X[col] = (
                X.groupby("Store")[col]
                 .transform(lambda s: s.ffill())
            )

            # Fill remaining NaNs using values learned during fit()
            missing = X[col].isna()

            X.loc[missing, col] = (
                X.loc[missing, "Store"]
                 .map(self.last_values_[col])
            )

            # Final fallback for unseen stores
            X[col] = X[col].fillna(self.global_values_[col])

        return X
