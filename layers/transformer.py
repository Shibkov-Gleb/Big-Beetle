"""Pre-normalized GPT decoder block."""

import tensorflow as tf
from layers.attention import CausalSelfAttention


@tf.keras.utils.register_keras_serializable(package="small_gpt")
class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.n_heads, self.d_ff = d_model, n_heads, d_ff
        self.dropout_rate = dropout
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.feed_forward = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation=tf.keras.activations.gelu),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout),
        ])

    def call(self, x, training=False):
        x = x + self.attention(self.norm1(x), training=training)
        return x + self.feed_forward(self.norm2(x), training=training)

    def get_config(self):
        config = super().get_config()
        config.update(d_model=self.d_model, n_heads=self.n_heads, d_ff=self.d_ff,
                      dropout=self.dropout_rate)
        return config

