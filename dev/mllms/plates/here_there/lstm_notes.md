### LSTM 
####  Single Cell Architecture
It's of 3 converging flows
1. Cell State       Ct
2. Hidden State     Ht
3. Input            Xt

--------------------------------------------------------------------------------------------------------------------------------
## LSTM Architecture

LSTM is a type of Recurrent Neural Network (RNN) designed to address the vanishing and exploding gradient problems faced by traditional RNNs when dealing with long-term dependencies. The key components of an LSTM cell are:

1. **Forget Gate**
2. **Input Gate**
3. **Output Gate**
4. **Cell State**

The internal working of an LSTM cell can be broken down into the following steps:

### Step 1: Forget Gate

The forget gate determines what information from the previous cell state should be forgotten or kept. It takes the previous hidden state `h(t-1)` and the current input `x(t)` as inputs and produces a output `f(t)` between 0 and 1 for each number in the cell state `C(t-1)`. A value close to 0 means "forget this", while a value close to 1 means "keep this".

```
f(t) = σ(W(f) * [h(t-1), x(t)] + b(f))
```

Here, `σ` is the sigmoid activation function, `W(f)` and `b(f)` are the weight and bias parameters for the forget gate, respectively.

### Step 2: Input Gate

The input gate decides what new information should be stored in the cell state. It consists of two parts: a sigmoid layer (`i(t)`) and a tanh layer (`g(t)`). The sigmoid layer determines which values will be updated, and the tanh layer creates a vector of new candidate values `g(t)` that could be added to the cell state.

```
i(t) = σ(W(i) * [h(t-1), x(t)] + b(i))
g(t) = tanh(W(g) * [h(t-1), x(t)] + b(g))
```

Here, `W(i)`, `b(i)`, `W(g)`, and `b(g)` are the weight and bias parameters for the input gate.

### Step 3: Cell State Update

The cell state `C(t)` is updated by combining the forget gate output `f(t)`, input gate output `i(t)`, and the new candidate values `g(t)`.

```
C(t) = f(t) * C(t-1) + i(t) * g(t)
```

This process allows the LSTM cell to selectively remember or forget information from the previous cell state and add new information to the current cell state.

### Step 4: Output Gate

The output gate decides what information from the current cell state should be output as the hidden state `h(t)`. It takes the current cell state `C(t)` and the previous hidden state `h(t-1)` as inputs and produces an output `o(t)` between 0 and 1.

```
o(t) = σ(W(o) * [h(t-1), x(t)] + b(o))
h(t) = o(t) * tanh(C(t))
```

Here, `W(o)` and `b(o)` are the weight and bias parameters for the output gate.

The final output `h(t)` is the hidden state at time `t`, which can be used as input for the next LSTM cell or as the output of the LSTM network.

## Additional Details

- **Peephole Connections**: LSTMs can also have peephole connections, which allow the cell state to directly influence the gates (forget, input, and output gates). This can improve the LSTM's ability to learn long-term dependencies.

- **Bidirectional LSTMs**: Bidirectional LSTMs (BiLSTMs) can process input sequences in both forward and backward directions, allowing them to capture context from both past and future information.

- **Stacked LSTMs**: Multiple LSTM layers can be stacked, with the output of one LSTM layer serving as input to the next layer, allowing the network to learn more complex representations.

- **Variations**: Several variations of LSTMs exist, such as Gated Recurrent Units (GRUs), which have a simpler architecture than LSTMs but can achieve similar performance in certain tasks.

LSTMs have been widely used in various applications, including natural language processing, speech recognition, machine translation, and time series forecasting, due to their ability to capture long-term dependencies and handle sequential data effectively.