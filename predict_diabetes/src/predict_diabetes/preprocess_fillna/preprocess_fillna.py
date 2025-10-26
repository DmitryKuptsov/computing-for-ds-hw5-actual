import pandas as pd


class FillNaPreprocessor:
    def __init__(self, columns=None):
        if columns is None:
            columns = ["height", "weight"]
        self.columns = columns

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        filled_df = df.copy()
        for col in self.columns:
            if col in filled_df.columns:
                mean_value = filled_df[col].mean(skipna=True)
                filled_df[col] = filled_df[col].fillna(mean_value)
        return filled_df