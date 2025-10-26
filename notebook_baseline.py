from predict_diabetes.src.predict_diabetes.data_loader import DataLoader
from predict_diabetes.src.predict_diabetes.preprocess_dropna import DropNaPreprocessor
from predict_diabetes.src.predict_diabetes.preprocess_fillna import FillNaPreprocessor
from predict_diabetes.src.predict_diabetes.feature_transformers import EthnicityFeature, GenderFeature
from predict_diabetes.src.predict_diabetes.model import Model
from sklearn.metrics import roc_auc_score


# 1. load data
loader = DataLoader("sample_diabetes_mellitus_data.csv")
train_df, test_df = loader.load_data()


# 2. preprocess data
dropna = DropNaPreprocessor()
fillna = FillNaPreprocessor()

train_df = dropna.transform(train_df)
test_df = dropna.transform(test_df)

train_df = fillna.transform(train_df)
test_df = fillna.transform(test_df)


# 3. feature transformations
ethnicity_feat = EthnicityFeature()
gender_feat = GenderFeature()

train_df = ethnicity_feat.transform(train_df)
test_df = ethnicity_feat.transform(test_df)

train_df = gender_feat.transform(train_df)
test_df = gender_feat.transform(test_df)


# 4. define model
numeric_non_null = train_df.select_dtypes(
    include=['int64', 'float64']
).dropna(axis=1)

numeric_non_null_new = numeric_non_null.columns.intersection(
    test_df.dropna(axis=1).columns
)

filtered_columns = (
    numeric_non_null_new[~numeric_non_null_new.str.endswith('_id')]
    .drop(['diabetes_mellitus', 'Unnamed: 0'])
)

ethnicity_columns = [
    col for col in train_df.columns if col.startswith("ethnicity_")
]

features = list(filtered_columns) + ethnicity_columns

target = "diabetes_mellitus"

model = Model(features, target, max_iter=1000)


# 5. train model
model.train(train_df)


# 6. predict on test set
test_df["predictions"] = model.predict(test_df)


# 7. evaluate
roc_auc = roc_auc_score(test_df[target], test_df["predictions"])
print(f"ROC-AUC on test set: {roc_auc:.3f}")