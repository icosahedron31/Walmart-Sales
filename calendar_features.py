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


    def transform(self, X):
        X = X.copy()
        dates = pd.to_datetime(X[self.date_col]).values.astype("datetime64[D]")
        X["year"] = dates.dt.year
        X["quarter"] = dates.dt.quarter
        X["month"] = dates.dt.month
        X["week"] = dates.dt.isocalendar().week.astype(int)

        return X

  