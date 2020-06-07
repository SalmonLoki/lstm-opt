import theano
import theano.tensor as T


class QuadraticFunction:
    def __init__(self, theta=None, W=None, b=None):
        self.W = W or T.matrix('W')
        self.b = b or T.vector('b')

        self.theta = theta or T.vector('theta')

        self.func = ((T.dot(self.W, self.theta) - self.b) ** 2).sum()
        self.grad = theano.grad(self.func, self.theta)

        self.params = [self.W, self.b]
