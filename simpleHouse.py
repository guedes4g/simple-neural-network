import numpy as np
np.random.seed(1)


X = np.array(([2, 9, 1], [1, 5, 1], [3, 6, 0], [3, 6, -1],  [1, 3, 2]), dtype=float)
y = np.array(([92], [86], [50], [100], [40]), dtype=float)

maxY = np.max(y)

X = X / np.amax(X, axis=0)
y = y / maxY


class NeuralNetwork(object):
    def __init__(self):
        # Hyperparemeters
        self.input_size = X.shape[1]
        self.hidden_size = X.shape[0]
        self.output_size = y.shape[1]

        # Weights for input -> hidden
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        # Weights for hidden -> output
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    def sigmoid(self, s):
        return 1/(1+np.exp(-s))

    def sigmoid_deriv(self, s):
        return s*(1-s)

    def forward(self, X):

        for i in range(10000):
            # Dot product of X (input) and first set of 3 x 2 weights
            z1 = np.dot(X, self.W1)
            # Applying the activation function
            o_z1 = self.sigmoid(z1)
            # Dot product of the hidden layer and second set of 3 x 1 weights
            z3 = np.dot(o_z1, self.W2)
            # Final activation function
            o_z3 = self.sigmoid(z3)
            
            # Calculate the error
            error_w2 =  y - o_z3
            delta_w2 = error_w2 * self.sigmoid_deriv(o_z3)

            error_w1 = delta_w2.dot(self.W2.T)
            delta_w1 = error_w1 * self.sigmoid_deriv(o_z1)

            # Update the Weight Values
            self.W2 += o_z1.dot(delta_w2)
            self.W1 += X.T.dot(delta_w1)

        print('error: ', np.mean(np.abs(error_w2)))
        return o_z3


brain = NeuralNetwork()

o = brain.forward(X)

print("Predicted result: "); print(o * maxY)
print("Actual output: "); print(y * maxY)
