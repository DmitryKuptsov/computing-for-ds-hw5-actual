from sklearn.linear_model import LogisticRegression


class Model:
    def __init__(self, features, target, **hyperparams):
        # private attributes
        self._features = features
        self._target = target
        self._hyperparams = hyperparams

        # public attribute — sklearn model instance
        self.model = LogisticRegression(**hyperparams)

    def train(self, df):
        X = df[self._features]
        y = df[self._target]
        self.model.fit(X, y)

    def predict(self, df):
        X = df[self._features]
        return self.model.predict_proba(X)[:, 1]