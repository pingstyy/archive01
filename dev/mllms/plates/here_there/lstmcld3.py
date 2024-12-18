import re
import numpy as np
import PyPDF2

# Constants
HIDDEN_SIZE = 256  # Size of the LSTM hidden layer
BATCH_SIZE = 32    # Batch size for training

# Helper functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# LSTM Cell
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

# LSTM Model
class LSTMModel:
    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # LSTM cell
        self.lstm_cell = LSTMCell(vocab_size, hidden_size)

        # Output layer
        self.W_o = np.random.randn(hidden_size, vocab_size)
        self.b_o = np.zeros(vocab_size)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        time_steps = inputs.shape[1]

        outputs = np.zeros((batch_size, time_steps, self.vocab_size))
        h_prev = np.zeros((batch_size, self.hidden_size))
        c_prev = np.zeros((batch_size, self.hidden_size))

        for t in range(time_steps):
            x_t = inputs[:, t]
            h_t, c_t = self.lstm_cell.forward(x_t, h_prev, c_prev)
            out_t = np.dot(h_t, self.W_o) + self.b_o  # Remove transpose operation
            outputs[:, t, :] = softmax(out_t)
            h_prev = h_t
            c_prev = c_t

        return outputs


    def predict(self, input_sequence):
        inputs = np.array([input_sequence])
        outputs = self.forward(inputs)[0]
        predicted_sequence = outputs.argmax(axis=-1)
        return predicted_sequence

    def save_components(self):
        np.save('embedding_layer.npy', self.embedding)
        np.save('output_weights.npy', self.W_o)
        np.save('output_biases.npy', self.b_o)
        np.save('lstm_weights.npy', [self.lstm_cell.W_f, self.lstm_cell.W_i, self.lstm_cell.W_o, self.lstm_cell.W_c])
        np.save('lstm_biases.npy', [self.lstm_cell.b_f, self.lstm_cell.b_i, self.lstm_cell.b_o, self.lstm_cell.b_c])

    def load_components(self):
        self.embedding = np.load('embedding_layer.npy')
        self.W_o = np.load('output_weights.npy')
        self.b_o = np.load('output_biases.npy')
        W_f, W_i, W_o, W_c = np.load('lstm_weights.npy', allow_pickle=True)
        b_f, b_i, b_o, b_c = np.load('lstm_biases.npy', allow_pickle=True)
        self.lstm_cell.W_f = W_f
        self.lstm_cell.W_i = W_i
        self.lstm_cell.W_o = W_o
        self.lstm_cell.W_c = W_c
        self.lstm_cell.b_f = b_f
        self.lstm_cell.b_i = b_i
        self.lstm_cell.b_o = b_o
        self.lstm_cell.b_c = b_c

# Data preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]+', '', text)  # Remove non-alphabetic characters
    words = text.split()
    word_to_idx = {word: idx for idx, word in enumerate(set(words))}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    vocab_size = len(word_to_idx)
    embedding_dim = min(vocab_size, 100)  # Set embedding dimension to minimum of vocab_size and 100
    sequences = []
    for i in range(len(words) - 120):  # Set sequence length to 120
        sequence = [word_to_idx[word] for word in words[i:i+121]]  # Include 121 words for input and target
        sequences.append(sequence)
    return sequences, word_to_idx, idx_to_word, vocab_size, embedding_dim

# Training function (modified)
def train(model, data, num_epochs, learning_rate):
    for epoch in range(num_epochs):
        # Shuffle the training data
        np.random.shuffle(data)

        # Split into batches
        num_batches = len(data) // BATCH_SIZE
        batches = np.array_split(data, num_batches)

        # Train on each batch
        for batch in batches:
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            # Forward pass
            outputs = model.forward(inputs)

            # Compute loss and gradients
            loss = -np.sum(targets * np.log(outputs)) / len(batch)
            # Backpropagation and update weights (not implemented)

        print(f"Epoch {epoch + 1}, Loss: {loss}")

# Evaluation function
def evaluate(model, data):
    inputs = data[:, :-1]
    targets = data[:, 1:]
    outputs = model.forward(inputs)
    loss = -np.sum(targets * np.log(outputs)) / len(data)
    return loss

# Command prompt interface
if __name__ == '__main__':
    while True:
        command = input("Enter command (train, test, predict, run, or exit): ")

        if command == "train":
            text_source = input("Enter text source (variable or file): ")
            if text_source == "variable":
                text = input("Enter text: ")
            elif text_source == "file":
                file_path = input("Enter file path: ")
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ''
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        page_text = re.sub(r'[^a-zA-Z0-9\s]+', ' ', page_text)
                        page_text = page_text.encode('utf-8', errors='ignore').decode('utf-8')
                        text += page_text
            else:
                print("Invalid text source. Please try again.")
                continue

            sequences, word_to_idx, idx_to_word, vocab_size, embedding_dim = preprocess_text(text)
            data = np.array(sequences)

            # Create model
            # Corrected instantiation of LSTMModel
            model = LSTMModel(vocab_size, HIDDEN_SIZE, use_embedding=True)

            num_epochs = int(input("Enter number of epochs: "))
            learning_rate = float(input("Enter learning rate: "))
            train(model, data, num_epochs, learning_rate)

            # Save model components
            model.save_components()

        elif command == "test":
            # Load model components
            model.load_components()

            test_loss = evaluate(model, data)
            print(f"Test Loss: {test_loss}")

        elif command == "predict":
            # Load model components
            model.load_components()

            input_sequence = input("Enter input sequence: ")
            input_sequence = [word_to_idx[word] for word in input_sequence.split()]
            predicted_sequence = model.predict(input_sequence)
            predicted_text = ' '.join([idx_to_word[idx] for idx in predicted_sequence])
            print(f"Predicted text: {predicted_text}")

        elif command == "run":
            # Load model components
            model.load_components()

            # Run the model with command line arguments
            text_source = input("Enter text source (variable or file): ")
            if text_source == "variable":
                text = input("Enter text: ")
            elif text_source == "file":
                file_path = input("Enter file path: ")
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ''
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        page_text = re.sub(r'[^a-zA-Z0-9\s]+', ' ', page_text)
                        page_text = page_text.encode('utf-8', errors='ignore').decode('utf-8')
                        text += page_text
            else:
                print("Invalid text source. Please try again.")
                continue

            sequences, word_to_idx, idx_to_word, vocab_size, embedding_dim = preprocess_text(text)
            data = np.array(sequences)

            input_sequence = input("Enter input sequence: ")
            input_sequence = [word_to_idx[word] for word in input_sequence.split()]
            predicted_sequence = model.predict(input_sequence)
            predicted_text = ' '.join([idx_to_word[idx] for idx in predicted_sequence])
            print(f"Predicted text: {predicted_text}")

        elif command == "exit":
            break

        else:
            print("Invalid command. Please try again.")