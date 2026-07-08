import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HolidayProximityTransformer(BaseEstimator, TransformerMixin):
    

    HOLIDAY_CALENDAR = {
        "Super Bowl": ["2009-02-06", "2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08", "2014-02-07"],
        "Labor Day": ["2009-09-04", "2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06", "2014-09-05"],
        "Thanksgiving": ["2009-11-27", "2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29", "2014-11-28"],
        "Christmas": ["2009-12-25", "2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27", "2014-12-26"],
    }

    def __init__(self, date_col="Date", max_weeks_cap=52, extra_holidays=None, add_cyclical=True):
        self.date_col = date_col
        self.max_weeks_cap = max_weeks_cap
      
        self.extra_holidays = extra_holidays
        self.add_cyclical = add_cyclical

    def fit(self, X, y=None):
        calendar = {k: list(v) for k, v in self.HOLIDAY_CALENDAR.items()}
        if self.extra_holidays:
            for name, dates in self.extra_holidays.items():
                calendar.setdefault(name, []).extend(dates)

        self.holiday_ts_ = {
            name: np.array(sorted(pd.to_datetime(dates)), dtype="datetime64[D]")
            for name, dates in calendar.items()
        }
        return self

    def _weeks_to_next(self, dates, holiday_ts):
        idx = np.searchsorted(holiday_ts, dates, side="left")
        idx_clipped = np.clip(idx, 0, len(holiday_ts) - 1)
        days = (holiday_ts[idx_clipped] - dates).astype("timedelta64[D]").astype(float)
        days[idx >= len(holiday_ts)] = np.nan  # date is after the last known occurrence
        return days / 7.0

    def _weeks_since_last(self, dates, holiday_ts):
        idx = np.searchsorted(holiday_ts, dates, side="left")
        idx_clipped = np.clip(idx - 1, 0, len(holiday_ts) - 1)
        days = (dates - holiday_ts[idx_clipped]).astype("timedelta64[D]").astype(float)
        days[idx == 0] = np.nan  # date is before the first known occurrence
        return days / 7.0

    def transform(self, X):
        X = X.copy()
        dates = pd.to_datetime(X[self.date_col]).values.astype("datetime64[D]")

        for name, holiday_ts in self.holiday_ts_.items():
            base = name.lower().replace(" ", "_")

            weeks_to = self._weeks_to_next(dates, holiday_ts)
            weeks_to_capped = np.clip(weeks_to, 0, self.max_weeks_cap)
            X[f"weeks_to_{base}"] = weeks_to_capped

            weeks_since = self._weeks_since_last(dates, holiday_ts)
            weeks_since_capped = np.clip(weeks_since, 0, self.max_weeks_cap)
            X[f"weeks_since_{base}"] = weeks_since_capped

            if self.add_cyclical:

                cycle_len = self.max_weeks_cap
                phase = np.where(
                    ~np.isnan(weeks_since),
                    weeks_since,
                    cycle_len - weeks_to,
                )
                angle = 2 * np.pi * phase / cycle_len
                X[f"{base}_phase_sin"] = np.sin(angle)
                X[f"{base}_phase_cos"] = np.cos(angle)

        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        holiday_cols = []
        for name in self.holiday_ts_.keys():
            b = name.lower().replace(" ", "_")
            holiday_cols += [f"weeks_to_{b}", f"weeks_since_{b}"]
            if self.add_cyclical:
                holiday_cols += [f"{b}_phase_sin", f"{b}_phase_cos"]
        return base + holiday_cols