class Loss(object):
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, y, yhat):
        raise NotImplementedError()

    def backward(self, y, yhat):
        raise NotImplementedError()
