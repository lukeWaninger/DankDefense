import lightgbm as lgb


class Model:
    def __init__(self, parameters):
        self.parameters = parameters
        self.model = None

    def train(self, x, y):
        for c in self.parmeters['categorical_features']:
            x[c] = x[c].astype('category')

        lgb_train = lgb.Dataset(x, y)
        self.model = lgb.train(self.parameters['params'], lgb_train, **self.parameters['kwargs'])

    def predict(self, x):
        return self.model.predict(x)

