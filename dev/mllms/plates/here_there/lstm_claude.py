import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Tanh activation function
def tanh(x):
    return np.tanh(x)

# LSTM cell
class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrices
        self.W_f = np.random.randn(hidden_size, hidden_size + input_size)
        self.W_i = np.random.randn(hidden_size, hidden_size + input_size)
        self.W_o = np.random.randn(hidden_size, hidden_size + input_size)
        self.W_c = np.random.randn(hidden_size, hidden_size + input_size)

        # Bias vectors
        self.b_f = np.zeros(hidden_size)
        self.b_i = np.zeros(hidden_size)
        self.b_o = np.zeros(hidden_size)
        self.b_c = np.zeros(hidden_size)

    def forward(self, x, h_prev, c_prev):
        combined = np.column_stack((h_prev, x))

        f_t = sigmoid(np.dot(combined, self.W_f.T) + self.b_f)
        i_t = sigmoid(np.dot(combined, self.W_i.T) + self.b_i)
        o_t = sigmoid(np.dot(combined, self.W_o.T) + self.b_o)
        c_tilde = tanh(np.dot(combined, self.W_c.T) + self.b_c)

        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * tanh(c_t)

        return h_t, c_t

# LSTM model
class LSTMModel:
    def __init__(self, input_size, hidden_size, output_size, sequence_length):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sequence_length = sequence_length

        self.lstm_cell = LSTMCell(input_size, hidden_size)
        self.W_y = np.random.randn(output_size, hidden_size)
        self.b_y = np.zeros(output_size)

    def forward(self, X):
        h_t = np.zeros(self.hidden_size)
        c_t = np.zeros(self.hidden_size)
        outputs = []

        for x_t in X:
            h_t, c_t = self.lstm_cell.forward(x_t, h_t, c_t)
            y_t = np.dot(self.W_y, h_t) + self.b_y
            outputs.append(y_t)

        return outputs

# Preprocess text data
def preprocess_data(text_corpus):
    # Tokenize the text corpus
    tokens = text_corpus.split()

    # Create a dictionary of unique tokens
    token_to_index = {token: idx for idx, token in enumerate(set(tokens))}
    index_to_token = {idx: token for token, idx in token_to_index.items()}

    # Convert tokens to indices
    input_sequences = []
    for i in range(len(tokens) - sequence_length):
        sequence = [token_to_index[token] for token in tokens[i:i+sequence_length]]
        input_sequences.append(sequence)

    return input_sequences, token_to_index, index_to_token

# Train the LSTM model
def train_lstm(input_sequences, token_to_index, index_to_token, epochs, batch_size):
    input_size = len(token_to_index)
    output_size = input_size
    sequence_length = len(input_sequences[0])

    model = LSTMModel(input_size, hidden_size, output_size, sequence_length)

    for epoch in range(epochs):
        # Shuffle input sequences
        np.random.shuffle(input_sequences)

        # Split input sequences into batches
        batches = [input_sequences[i:i+batch_size] for i in range(0, len(input_sequences), batch_size)]

        for batch in batches:
            # Convert input sequences to one-hot encoded vectors
            batch_inputs = np.zeros((len(batch), sequence_length, input_size))
            for i, sequence in enumerate(batch):
                for j, token_index in enumerate(sequence):
                    batch_inputs[i, j, token_index] = 1

            # Forward pass
            outputs = model.forward(batch_inputs)

            # Compute loss and update weights (not shown)
            # ...

    return model

# Use the trained LSTM model for text autocomplete
def autocomplete_text(model, seed_text, token_to_index, index_to_token, max_length):
    input_size = len(token_to_index)
    sequence_length = model.sequence_length

    # Convert seed text to input sequence
    seed_tokens = seed_text.split()
    input_sequence = [token_to_index[token] for token in seed_tokens[-sequence_length:]]
    input_sequence = np.array(input_sequence).reshape(1, sequence_length)

    # Prepare input for LSTM
    input_vector = np.zeros((1, sequence_length, input_size))
    for i, token_index in enumerate(input_sequence[0]):
        input_vector[0, i, token_index] = 1

    # Generate text
    output_text = seed_text
    for _ in range(max_length):
        output = model.forward(input_vector)
        next_token_probs = output[-1].ravel()
        next_token_index = np.random.choice(input_size, p=next_token_probs/np.sum(next_token_probs))
        next_token = index_to_token[next_token_index]

        output_text += ' ' + next_token

        # Update input sequence
        input_sequence = np.roll(input_sequence, -1, axis=1)
        input_sequence[0, -1] = next_token_index
        input_vector = np.zeros((1, sequence_length, input_size))
        for i, token_index in enumerate(input_sequence[0]):
            input_vector[0, i, token_index] = 1

    return output_text

# Example usage
text_corpus = "This is a sample text corpus for demonstrating the LSTM text autocomplete model."
sequence_length = 3
hidden_size = 128
epochs = 10
batch_size = 32

input_sequences, token_to_index, index_to_token = preprocess_data(text_corpus)
model = train_lstm(input_sequences, token_to_index, index_to_token, epochs, batch_size)

seed_text = "This is"
max_length = 20
autocompleted_text = autocomplete_text(model, seed_text, token_to_index, index_to_token, max_length)
print(autocompleted_text)