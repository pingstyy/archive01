import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample text data
text_data = [
    "The quick brown fox jumps over the lazy dog",
    "The quick brown fox jumps over the lazy cat",
    "The quick brown dog jumps over the lazy cat",
    "The quick brown dog jumps over the lazy fox"
]

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
vocab_size = len(tokenizer.word_index) + 1

# Convert text data to sequences
sequences = tokenizer.texts_to_sequences(text_data)
max_sequence_length = max([len(seq) for seq in sequences])

# Pad sequences to ensure uniform length
sequences_padded = pad_sequences(sequences, maxlen=max_sequence_length, padding='pre')

# Prepare input and output sequences for training
X = sequences_padded[:, :-1]
y = sequences_padded[:, -1]

# Define the LSTM model
embedding_dim = 50
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_sequence_length-1),
    LSTM(100),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
model.fit(X, y, epochs=100, verbose=1)

# Function to generate text predictions
def generate_text(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Generate text suggestions
seed_text = "The quick brown"
predicted_text = generate_text(seed_text, 3)
print("Predicted text:", predicted_text)
