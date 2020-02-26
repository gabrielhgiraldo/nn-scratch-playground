import numpy as np


class ANN:
    class Layer:
        def __init__(self, shape: tuple, activation='relu'):
            self.shape = shape
            self.W = np.random.rand(*self.shape)
            self.B = np.zeros((self.shape[0], 1))
            self.activation = self.relu if activation == 'relu' else self.sigmoid
            self.activation_prime = self.relu_prime if activation == 'relu' else self.sigmoid_prime
            self.A = None
            self.Z = None

        def relu(self, Z):
            # element-wise matrix multiplication by boolean value of whether each element in X is > 0 (a.k.a relu)
            return Z*(Z > 0)

        def relu_prime(self, Z):
            return (Z > 0).astype(int)

        def sigmoid(self, Z):
            return 1/(1 + np.exp(-Z))

        def sigmoid_prime(self, Z):
            return self.sigmoid(Z) * (1 - self.sigmoid(Z))

        def activate(self, a_prev):
            self.Z = np.dot(self.W, a_prev) + self.B
            self.A = self.activation(self.Z)
            return self.A

        def activate_backwards(self, Z):
            return self.activation_prime(Z)

        def update(self, dW, dB, learning_rate):
            self.W -= learning_rate * dW
            self.B -= learning_rate * dB

    def __init__(self, input_shape=(100,1), n_layers=1, layer_dims=None, learning_rate=0.01, n_classes=2):
        self.input_shape = input_shape
        self.learning_rate = learning_rate

        #input layer
        self.layers = [self.Layer((np.random.randint(1, 5), self.input_shape[0]))]
        #hidden layers
        for l in range(n_layers):
            n_nodes = np.random.randint(2, 5)
            n_prev_nodes = self.layers[l].shape[0]
            layer_shape = (n_nodes, n_prev_nodes)
            self.layers.append(self.Layer(shape=layer_shape))
        #output layer
        n_nodes = n_classes if n_classes > 2 else 1
        n_prev_nodes = self.layers[-1].shape[0]
        self.output_shape = (n_nodes, n_prev_nodes)
        self.layers.append(self.Layer(
            shape=self.output_shape, activation='sigmoid'))

    def calculate_cost(self, Y_pred, Y_true):
        N = Y_pred.shape[0]
        return -(1/N)*np.sum(Y_true*np.log(Y_pred) + (1-Y_true)*np.log(1-Y_pred))

    def forward_pass(self, X):
        A = np.array(X).reshape(self.input_shape)
        for layer in self.layers:
            A = layer.activate(A)
        return A

    def backward_pass(self, Y_pred, Y_true, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
        cost = self.calculate_cost(Y_pred, Y_true)
        dA = (Y_true/Y_pred) + ((1-Y_true)/(1-Y_pred))
        backward_layers = reversed(list(enumerate(self.layers)))
        for l_prev, layer in backward_layers:
            M = self.layers[l_prev].A.shape[1]
            dZ = layer.activate_backwards(layer.Z)
            dW = np.dot(self.layers[l_prev].A.T, dA*dZ)
            dB = np.sum(dZ, axis=1, keepdims=True)/M
            layer.update(dW, dB, learning_rate)
            dA = np.dot(layer.W.T, dA*dZ)
        return cost

    def fit(self, X_train, y_train, num_epochs=10, iterations=10, learning_rate=None):
        batch_size = self.input_shape[0]
        epoch = 0
        #TODO: FIX BATCHING
        for i in range(num_epochs):
            for iteration in range(iterations):
                x = X_train[i*batch_size:(i+1)*batch_size]
                y_true = y_train[i*batch_size:(i+1)*batch_size]
                Y_pred = self.forward_pass(x)
                cost = self.backward_pass(Y_pred, y_true)
                epoch += 1
                if iteration % 10 == 0:
                    print(f'epoch:{epoch}: iteration {iteration}: cost:{cost}')
                if epoch == num_epochs:
                    return cost
        return cost
