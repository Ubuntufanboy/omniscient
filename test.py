import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import datacollector
import numpy as np
import dataset
import functools
from collections import defaultdict, Counter
import data

# Transformer block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.03):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Model definition
def create_transformer_model(seq_len, num_features, embed_dim, num_heads, ff_dim, num_classes):
    inputs = Input(shape=(seq_len, num_features))
    # Map input features to the embedding dimension
    x = Dense(embed_dim)(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x, training=True)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def mod(num, divisor):
    return int(num) % divisor 
    # This seems pointless but TF tries to do FORMATTING with the % operator. Guess how many hours that stumped me!

def preprocess_number_slow(number):
    number = int(number)
    features = []
    features.append(number / 100)  # The number itself
    features.append(mod(number, 2) / 2)  # Even or odd
    features.append(mod(number, 3) / 3)  # Divisibility by 3
    features.append(mod(number,5) / 5)  # Divisibility by 5
    features.append(mod(number,10) / 9)  # Divisibility by 10
    features.append(len([i for i in range(1, number + 1) if mod(number, i) == 0]) / 12)  # Number of divisors
    features.append(sum(int(digit) for digit in str(number)) / 18)  # Digit sum
    features.append((dataset.dataset.count(number))/ 100) # Rough ranking in occurence
    features.append(max([int(num) for num in list(str(number))]) / 9)
    features.append(min([int(num) for num in list(str(number))]) / 9)
    features.append(int(max(list(str(number)))) - int(min(list(str(number)))) / 100)

    if number == 100:
        features.append(1) # 9 / 9
        features.append(1) # 9 / 9
        ## 100 is basically 99 in terms of a human brain not 1-0-0
    else:
        if number >= 10 and number != 100:
            features.append(int(str(number)[0]) / 9)
        else:
            features.append(0)
        features.append(int(str(number)[-1]) / 9)
    return features

@functools.lru_cache(maxsize=128)
def preprocess_number(number):
    return preprocess_number_slow(number)

# Hyperparameters
seq_len = 20  # Define according to your sequence length
num_features = 27  # Number of features
embed_dim = 32  # Embedding size for each token
num_heads = 8  # Number of attention heads
ff_dim = 64  # Hidden layer size in feed forward network inside transformer
num_classes = 100  # Number of classes

# Create and compile the model
model = create_transformer_model(seq_len, num_features, embed_dim, num_heads, ff_dim, num_classes)

model.load_weights(f"godly.weights.h5")

def run907test(X_train, Y_train):
    correct = 0
    X = np.array([
    [
        preprocess_number(num) +
        preprocess_number(sequence[i-1] if i > 0 else 37) +
        [(num - (sequence[i-1] if i > 0 else 37)) / 99]
    for i, num in enumerate(sequence)]
for sequence in X_train])
    Y_train = [int(num) - 1 for num in Y_train]
    y = to_categorical(Y_train, num_classes=num_classes)
    predicted_logits = model.predict(X)
    for i in range(len(predicted_logits)):
        guess = np.argmax(predicted_logits[i])
        if guess == np.argmax(y[i]):
            correct += 1
            #print(f"Correct prediction: Correctly guessed {guess}")
        else:
            #print(f"Incorrect prediction: Expected {np.argmax(y[i])}, got {guess}")
            pass
    return correct / len(predicted_logits)
X = datacollector.getx()
Y = datacollector.gety()
print(f"Got {run907test(X, Y)*100}% Accuracy!")

