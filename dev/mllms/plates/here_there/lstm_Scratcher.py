import numpy as np

# Define sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define derivative of sigmoid activation function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define LSTM class
class LSTM:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.bf = np.zeros((hidden_size, 1))
        self.wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.bi = np.zeros((hidden_size, 1))
        self.wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.bc = np.zeros((hidden_size, 1))
        self.wo = np.random.randn(hidden_size, input_size + hidden_size)
        self.bo = np.zeros((hidden_size, 1))
        self.wy = np.random.randn(output_size, hidden_size)
        self.by = np.zeros((output_size, 1))
        
    def forward(self, x, h_prev, c_prev):
        self.x = x
        self.h_prev = h_prev
        self.c_prev = c_prev
        
        # Concatenate input and previous hidden state
        concat_input = np.concatenate((h_prev, x), axis=0)
        
        # Forget gate
        self.f = sigmoid(np.dot(self.wf, concat_input) + self.bf)
        
        # Input gate
        self.i = sigmoid(np.dot(self.wi, concat_input) + self.bi)
        
        # Candidate cell state
        self.c_hat = np.tanh(np.dot(self.wc, concat_input) + self.bc)
        
        # Update cell state
        self.c_next = self.f * c_prev + self.i * self.c_hat
        
        # Output gate
        self.o = sigmoid(np.dot(self.wo, concat_input) + self.bo)
        
        # Next hidden state
        self.h_next = self.o * np.tanh(self.c_next)
        
        # Output prediction
        self.y_pred = np.dot(self.wy, self.h_next) + self.by
        return self.y_pred, self.h_next, self.c_next
    
    def backward(self, y, y_pred, dh_next, dc_next):
        # Compute gradients
        dy = y_pred - y
        dwy = np.dot(dy, self.h_next.T)
        dby = dy
        dh = np.dot(self.wy.T, dy) + dh_next
        do = dh * np.tanh(self.c_next)
        do = sigmoid_derivative(self.o) * do
        dwo = np.dot(do, np.concatenate((self.h_prev, self.x), axis=0).T)
        dbo = do
        dc = dh * self.o * (1 - np.tanh(self.c_next)**2) + dc_next
        dc_hat = dc * self.i
        dc_hat = dc_hat * (1 - self.c_hat**2)
        dwc = np.dot(dc_hat, np.concatenate((self.h_prev, self.x), axis=0).T)
        dbc = dc_hat
        di = dc * self.c_hat
        di = sigmoid_derivative(self.i) * di
        dwi = np.dot(di, np.concatenate((self.h_prev, self.x), axis=0).T)
        dbi = di
        df = dc * self.c_prev
        df = sigmoid_derivative(self.f) * df
        dwf = np.dot(df, np.concatenate((self.h_prev, self.x), axis=0).T)
        dbf = df
        
        # Update weights and biases
        self.wy -= self.learning_rate * dwy
        self.by -= self.learning_rate * dby
        self.wf -= self.learning_rate * dwf
        self.bf -= self.learning_rate * dbf
        self.wi -= self.learning_rate * dwi
        self.bi -= self.learning_rate * dbi
        self.wc -= self.learning_rate * dwc
        self.bc -= self.learning_rate * dbc
        self.wo -= self.learning_rate * dwo
        self.bo -= self.learning_rate * dbo
        
        return dh[:self.hidden_size], dc[:self.hidden_size]
    
    def train(self, inputs, targets, num_iterations):
        for i in range(num_iterations):
            for j in range(len(inputs)):
                x = inputs[j]
                y = targets[j]
                
                # Forward pass
                y_pred, self.h_next, self.c_next = self.forward(x, self.h_prev, self.c_prev)
                
                # Backward pass
                dh_next, dc_next = self.backward(y, y_pred, self.h_next, self.c_next)
                
                # Update previous hidden state and cell state
                self.h_prev = self.h_next
                self.c_prev = self.c_next
                
                # Reset previous hidden state and cell state for next sequence
                self.h_prev = np.zeros_like(self.h_prev)
                self.c_prev = np.zeros_like(self.c_prev)
                
# Test the LSTM model
inputs = [np.array([[1], [2], [3]]), np.array([[2], [3], [4]]), np.array([[3], [4], [5]])]
targets = [4, 5, 6]

lstm = LSTM(input_size=1, hidden_size=2, output_size=1, learning_rate=0.1)
lstm.train(inputs, targets, num_iterations=1000)

# Generate predictions
x_test = np.array([[4], [5], [6]])
y_pred, _, _ = lstm.forward(x_test, np.zeros((2, 1)), np.zeros((2, 1)))
print("Predicted value:", y_pred[0][0])



