"""Masked multi-head self-attention for an autoregressive decoder."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="small_gpt")
class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim = d_model // n_heads
        self.dropout_rate = dropout
        self.qkv = tf.keras.layers.Dense(3 * d_model)
        self.projection = tf.keras.layers.Dense(d_model)
        self.attention_dropout = tf.keras.layers.Dropout(dropout)
        self.output_dropout = tf.keras.layers.Dropout(dropout)

    def _split_heads(self, x):
        batch, length = tf.shape(x)[0], tf.shape(x)[1]
        x = tf.reshape(x, (batch, length, self.n_heads, self.head_dim))
        return tf.transpose(x, (0, 2, 1, 3))

    def call(self, x, training=False):
        query, key, value = tf.split(self.qkv(x), 3, axis=-1)
        query, key, value = map(self._split_heads, (query, key, value))
        scale = tf.cast(self.head_dim, x.dtype) ** -0.5
        scores = tf.matmul(query, key, transpose_b=True) * scale

        length = tf.shape(x)[1]
        causal = tf.linalg.band_part(tf.ones((length, length), tf.bool), -1, 0)
        scores = tf.where(causal[None, None, :, :], scores,
                          tf.cast(-1e4, scores.dtype))
        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.attention_dropout(weights, training=training)
        context = tf.matmul(weights, value)
        context = tf.transpose(context, (0, 2, 1, 3))
        context = tf.reshape(context, (tf.shape(x)[0], length, self.d_model))
        return self.output_dropout(self.projection(context), training=training)

    def get_config(self):
        config = super().get_config()
        config.update(d_model=self.d_model, n_heads=self.n_heads,
                      dropout=self.dropout_rate)
        return config