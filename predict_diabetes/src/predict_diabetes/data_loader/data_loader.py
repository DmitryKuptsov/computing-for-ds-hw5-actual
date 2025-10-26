import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_data(self, test_size: float = 0.2, random_state: int = 42):
        try:
            df = pd.read_csv(self.filepath)
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
            return train_df, test_df
        except FileNotFoundError: #dealing with exceptions
            print(f"File not found: {self.filepath}")
            return pd.DataFrame(), pd.DataFrame()
        except Exception as e:
            print(f"Error while loading data: {e}")
            return pd.DataFrame(), pd.DataFrame()