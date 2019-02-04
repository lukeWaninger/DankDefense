import lightgbm as lgb


class Model:
    def __init__(self, parameters):
        self.parameters = parameters
        self.model = None

        if 'kwargs' not in self.parameters['params'].keys():
            self.kwargs = {}
        else:
            self.kwargs = self.parameters['params']['kwargs']
            del self.parameters['params']['kwargs']

    def train(self, x, y):
        lgb_train = lgb.Dataset(x, y)

        with open('a_small_demo_log.txt', 'w') as f:
            f.write(self.kwargs)

        self.model = lgb.train(self.parameters['params'], lgb_train, **self.kwargs)

    def predict(self, x):
        return self.model.predict(x)

