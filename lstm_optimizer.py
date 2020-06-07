import numpy as np
import theano
import theano.tensor as T
import lasagne as L
import quadratic_function
import lstm_layer as lstm

input_var = T.vector()


def build_net(func, n_steps, n_hidden=20, gradient_steps=-1):
    l_input = L.layers.InputLayer(shape=(None,), input_var=input_var)
    l_optim = lstm.lstm_layer(l_input,
                              num_units=n_hidden,
                              n_steps=n_steps,
                              function=func,
                              gradient_steps=gradient_steps)
    return l_optim


class lstm_ptimizer:
    def __init__(self, input_var, func, func_params, n_hidden=20, gradient_steps=20, n_gac=0):
        n_steps = T.iscalar()
        self.l_optim = build_net(func, n_steps, n_hidden, gradient_steps=gradient_steps)

        theta_history, loss_history = L.layers.get_output(self.l_optim)
        loss = loss_history.sum()

        self.lr = theano.shared(np.array(0.01, dtype=np.float32))

        params = L.layers.get_all_params(self.l_optim)
        updates = L.updates.adam(loss, params, learning_rate=self.lr)

        self.loss_fn = theano.function([input_var, n_steps] + func_params, [theta_history, loss_history],
                                       allow_input_downcast=True)
        self.train_fn = theano.function([input_var, n_steps] + func_params, [theta_history, loss_history],
                                        updates=updates, allow_input_downcast=True)

    def train(self, sample_function, sample_point, n_iter=100, n_epochs=50, batch_size=100, decay_rate=0.96):
        for i in range(n_epochs):
            training_loss_history = []
            for j in range(batch_size):
                params = sample_function()
                theta = sample_point()
                theta_history, loss_history = self.train_fn(theta, n_iter, *params)
                training_loss_history.append(loss_history)
            self.lr.set_value((self.lr.get_value() * decay_rate).astype(np.float32))

    def optimize(self, theta, func_params, n_iter):
        return self.loss_fn(theta, n_iter, *func_params)


W = T.matrix()
b = T.vector()
func = lambda theta: quadratic_function.QuadraticFunction(theta, W, b).func


