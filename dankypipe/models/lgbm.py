import lightgbm as lgb


class Model:
    def __init__(self, parameters):
        self.parameters = parameters
        self.model = None

        if 'kwargs' not in self.parameters:
            self.parameters['kwargs'] = {}

    def train(self, x, y):
        lgb_train = lgb.Dataset(x, y)

        self.model = lgb.train(self.parameters['params'], lgb_train, **self.parameters.kwargs)

    def predict(self, x):
        return self.model.predict(x)

