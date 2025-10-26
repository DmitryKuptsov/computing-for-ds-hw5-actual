import pandas as pd


class DropNaPreprocessor:
    def __init__(self, columns=None):
        if columns is None:
            columns = ["age", "gender", "ethnicity"]
        self.columns = columns

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned_df = df.dropna(subset=self.columns)
        return cleaned_df