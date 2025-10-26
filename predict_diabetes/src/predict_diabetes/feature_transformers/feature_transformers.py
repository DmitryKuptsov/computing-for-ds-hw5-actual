from abc import ABC, abstractmethod
import pandas as pd


class BaseFeature(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class EthnicityFeature(BaseFeature):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = pd.get_dummies(df, columns=["ethnicity"], drop_first=True)
        return df_encoded


class GenderFeature(BaseFeature):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy["gender"] = df_copy["gender"].map(lambda x: 1 if str(x).upper().startswith("M") else 0)
        return df_copy


# comment:
# alternatively, it could be implemented with a concrete parent class
# that already defines the one-hot encoding logic,
# and EthnicityFeature / GenderFeature could simply inherit from it
# specifying the target column.
# here, an abstract BaseFeature class is kept as required by the assignment
# to demonstrate the concept of polymorphism
