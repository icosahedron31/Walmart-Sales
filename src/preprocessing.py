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
# ============ step 3: the forecaster (final estimator) ============
from neuralforecast.losses.pytorch import MAE
MAX_STEPS=10000
SEED=42
H = 35
INPUT_SIZE=52
FREQ = 'W-FRI'
class PatchTSTForecaster(BaseEstimator):
    """fit(): trains on the gap-filled panel and KEEPS it as history.
       predict(): takes test keys, forecasts from that stored history, merges."""

    def __init__(self, h=H, input_size=INPUT_SIZE, patch_len=8, stride=4,
                 hidden_size=128, n_heads=8, encoder_layers=3, dropout=0.2,
                 learning_rate=1e-3, max_steps=MAX_STEPS, freq=FREQ, seed=SEED):
        self.h, self.input_size = h, input_size
        self.patch_len, self.stride = patch_len, stride
        self.hidden_size, self.n_heads = hidden_size, n_heads
        self.encoder_layers, self.dropout = encoder_layers, dropout
        self.learning_rate, self.max_steps = learning_rate, max_steps
        self.freq, self.seed = freq, seed

    def fit(self, X, y=None):
        model = PatchTST(
            h=self.h, input_size=self.input_size,
            patch_len=self.patch_len, stride=self.stride,
            hidden_size=self.hidden_size, n_heads=self.n_heads,
            encoder_layers=self.encoder_layers, dropout=self.dropout,
            revin=True,                  # instance norm — the core of the architecture
            scaler_type="identity",      # revin already normalizes; don't stack them
            loss=MAE(), learning_rate=self.learning_rate,
            max_steps=self.max_steps, batch_size=256, random_seed=self.seed,
        )
        self.nf_ = NeuralForecast(models=[model], freq=self.freq)
        self.nf_.fit(X, val_size=0)
        self.history_ = X               # frozen. predict() never re-derives it.
        return self

    def predict(self, X):
        # X = test keys, straight out of GapFiller's passthrough
        P = self.nf_.predict(df=self.history_).reset_index()

        out = X.merge(P, on=["unique_id", "ds"], how="left")
        out["Weekly_Sales"] = out["PatchTST"].fillna(0.0)   # test-only depts: no history
        out["Id"] = out["unique_id"] + "_" + out["ds"].dt.strftime("%Y-%m-%d")
        return out[["Id", "Weekly_Sales"]]