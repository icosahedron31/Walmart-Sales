import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import BaseEstimator, TransformerMixin
from utilsforecast.preprocessing import fill_gaps



from sklearn.base import BaseEstimator, TransformerMixin

class DateToTimeIdx(BaseEstimator, TransformerMixin):

    def __init__(self, date_col='Date', time_idx_col='time_idx'):
        self.date_col = date_col
        self.time_idx_col = time_idx_col

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        dates = pd.to_datetime(X[self.date_col])
        self.min_date_ = dates.min()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col])

        delta_days = (X[self.date_col] - self.min_date_).dt.days
        X[self.time_idx_col] = (delta_days // 7).astype(int)

        return X
class PickColumns(BaseEstimator, TransformerMixin):

    def __init__(self, columns_to_keep):
        self.columns_to_keep = columns_to_keep

    def fit(self, X, y=None):

        return self

    def transform(self, X):


        X = pd.DataFrame(X).copy()
        X["unique_id"] = X["Store"].astype(str) + "_" + X["Dept"].astype(str)
        X["ds"] = X["Date"]
        X["y"] = X["Weekly_Sales"]
        X = X[self.columns_to_keep]
        X = X.sort_values(["unique_id", "ds"])
        return X
# ============ step 2: gap filling ============
FREQ = 'W-FRI'
class GapFiller(BaseEstimator, TransformerMixin):
    """Rectangular grid + zero-fill + WMAE-weighted mask.

    No y (raw test) -> passthrough. Test rows carry no history; they're just
    the (unique_id, ds) pairs we must submit.
    """

    def __init__(self, freq=FREQ):
        self.freq = freq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if "y" not in X.columns:
            return X
        holiday = X.drop_duplicates("ds").set_index("ds")["IsHoliday_x"]

        d = fill_gaps(X[["unique_id", "ds", "y"]], freq=self.freq,
                      start="global", end="global")

        observed = d["y"].notna()        # AFTER the fill: NaN == injected row
        d["y"] = d["y"].fillna(0.0)      # absent row == no sales. never interpolate.

        hol = d["ds"].map(holiday).fillna(False)

        # 0 = injected (fabricated, not scored) | 5 = holiday | 1 = normal
        d["available_mask"] = np.where(~observed, 0.0, np.where(hol, 5.0, 1.0))

        assert d.groupby("unique_id").size().nunique() == 1, "not rectangular"
        return d.sort_values(["unique_id", "ds"]).reset_index(drop=True)

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ConvertToCategorical(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns or ['Store', 'Dept', 'IsHoliday_x', 'Type']

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(self.columns)

        missing = set(self.columns_) - set(X.columns)
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")

        self.categories_ = {
            col: pd.Series(X[col].astype(str).unique()).sort_values().tolist()
            for col in self.columns_
        }
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in self.columns_:
            X[col] = pd.Categorical(X[col].astype(str), categories=self.categories_[col])
        return X
    
class WeightMetric(BaseEstimator, TransformerMixin):


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["wmae_weight"] = np.where(X["IsHoliday_x"], 5.0, 1.0)

        return X
class DropColumn(BaseEstimator, TransformerMixin):

    def __init__(self, drop_cols=None):
        self.drop_cols = drop_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X.drop(columns=self.drop_cols, inplace=True)

        return X
