import lightgbm as lgb


class Model:
    def __init__(self, parameters, kwargs=None):
        self.parameters = parameters
        self.model = None
        self.kwargs = kwargs if kwargs is not None else {}

    def train(self, x, y):
        lgb_train = lgb.Dataset(x, y)

        with open('a_small_demo_log.txt', 'w') as f:
            f.write(self.kwargs)

        self.model = lgb.train(self.parameters['params'], lgb_train, **self.kwargs)

    def predict(self, x):
        return self.model.predict(x)

