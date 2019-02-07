import lightgbm as lgb


class Model:
    def __init__(self, parameters):
        self.parameters = parameters
        self.model = None

    def train(self, x, y):
        print(self.parameters)
        for c in self.parameters['categorical_features']:
            x.loc[:, c] = x[c].astype('category')

        lgb_train = lgb.Dataset(x, y['Target'])
        self.model = lgb.train(self.parameters['params'], lgb_train, **self.parameters['kwargs'])

    def predict(self, x):
        for c in self.parameters['categorical_features']:
            x.loc[:, c] = x[c].astype('category')

        return self.model.predict(x)

