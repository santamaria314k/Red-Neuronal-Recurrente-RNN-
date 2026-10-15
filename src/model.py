"""LSTM bidireccional con vectorización de texto integrada en el modelo."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config


def build_model(train_texts) -> keras.Model:
    """El vocabulario se aprende SOLO con el conjunto de entrenamiento (sin fuga de datos).
    Como TextVectorization va dentro del modelo, la app le pasa texto crudo."""
    vectorizer = layers.TextVectorization(
        max_tokens=config.MAX_TOKENS, output_mode="int",
        output_sequence_length=config.SEQ_LEN, name="vectorizacion")
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(tf.constant(train_texts)).batch(256))

    inputs = keras.Input(shape=(), dtype="string", name="texto")
    x = vectorizer(inputs)
    x = layers.Embedding(config.MAX_TOKENS, config.EMBED_DIM, mask_zero=True, name="embedding")(x)
    x = layers.Bidirectional(layers.LSTM(config.LSTM_UNITS, dropout=0.2), name="bilstm")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prob_positivo")(x)
    model = keras.Model(inputs, outputs, name="rnn_sentimiento")
    model.compile(
        optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model
