"""Token embeddings plus GPT-2-style learned positional embeddings."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="small_gpt")
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.position_embedding = tf.keras.layers.Embedding(max_seq_len, d_model)

    def call(self, token_ids):
        length = tf.shape(token_ids)[1]
        tf.debugging.assert_less_equal(length, self.max_seq_len)
        positions = tf.range(length)[tf.newaxis, :]
        return self.token_embedding(token_ids) + self.position_embedding(positions)

    @property
    def token_weights(self):
        return self.token_embedding.embeddings

    def get_config(self):
        config = super().get_config()
        config.update(vocab_size=self.vocab_size, max_seq_len=self.max_seq_len,
                      d_model=self.d_model)
        return config

