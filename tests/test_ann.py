import unittest

import numpy as np

from ANN import ANN

class TestANN(unittest.TestCase):
    def test_layer_dims(self):
        ann = ANN(n_layers=3)
        for n, layer in enumerate(ann.layers[1:]):
            prev_layer = ann.layers[n]
            with self.subTest(prev_layer=prev_layer, layer=layer):
                self.assertEqual(prev_layer.shape[0], layer.shape[1])

    def test_forward_pass(self):
        batch_size = 100
        input_shape = (10,batch_size)
        ann = ANN(input_shape=input_shape)
        X = np.random.rand(*input_shape)
        Y_pred = ann.forward_pass(X).T
        self.assertEqual(Y_pred.shape, (batch_size,1))
    
    def test_backward_pass(self):
        batch_size = 100
        input_shape = (10, batch_size)
        ann = ANN(input_shape=input_shape)
        X_train = np.random.rand(*input_shape)
        y_train = np.random.randint(1, size=(batch_size,1))
        Y_pred = ann.forward_pass(X_train)
        cost = ann.backward_pass(Y_pred, y_train)
        self.assertIsInstance(cost, float)
        self.assertGreater(cost, 0)

    def test_fit(self):
        input_shape = (10,1)
        n_samples = 1000
        ann = ANN(input_shape=input_shape)
        X_train = np.random.rand(n_samples, *input_shape)
        y_train = np.random.randint(1, size=(n_samples,1,1))
        cost = ann.fit(X_train, y_train)
        self.assertGreater(cost, 0)

    # def test_mnist(self):
        

if __name__ == "__main__":
    unittest.main()



        