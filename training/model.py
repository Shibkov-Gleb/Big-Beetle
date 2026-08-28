"""Compact GPT language model with tied input/output embeddings."""

import tensorflow as tf

from config import GPTConfig
from data_preprocessing.embeddings import TokenAndPositionEmbedding
from layers.transformer import TransformerDecoderBlock


@tf.keras.utils.register_keras_serializable(package="small_gpt")
class GPTModel(tf.keras.Model):
    def __init__(self, config: GPTConfig, **kwargs):
        super().__init__(**kwargs)
        self.gpt_config = config
        self.embedding = TokenAndPositionEmbedding(
            config.vocab_size, config.max_seq_len, config.d_model)
        self.embedding_dropout = tf.keras.layers.Dropout(config.dropout)
        self.blocks = [TransformerDecoderBlock(
            config.d_model, config.n_heads, config.d_ff, config.dropout,
            name=f"decoder_{i}") for i in range(config.n_layers)]
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, token_ids, training=False):
        x = self.embedding_dropout(self.embedding(token_ids), training=training)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.final_norm(x)
        return tf.einsum("btd,vd->btv", x, self.embedding.token_weights)

    def generate(self, token_ids, max_new_tokens=80, temperature=0.8,
                 top_k=40, eos_id=50_256):
        ids = tf.convert_to_tensor(token_ids, dtype=tf.int32)
        for _ in range(max_new_tokens):
            context = ids[:, -self.gpt_config.max_seq_len:]
            logits = self(context, training=False)[:, -1, :]
            if temperature <= 0:
                next_id = tf.argmax(logits, axis=-1, output_type=tf.int32)[:, None]
            else:
                logits = logits / temperature
                if top_k:
                    values, indices = tf.math.top_k(logits, k=top_k)
                    sample = tf.random.categorical(values, 1)
                    next_id = tf.gather(indices, sample, batch_dims=1)
                else:
                    next_id = tf.random.categorical(logits, 1, dtype=tf.int32)
            ids = tf.concat([ids, next_id], axis=1)
            if bool(tf.reduce_all(next_id == eos_id)):
                break
        return ids

    def get_config(self):
        return {"config": self.gpt_config.__dict__, **super().get_config()}

    @classmethod
    def from_config(cls, config):
        model_config = GPTConfig(**config.pop("config"))
        return cls(model_config, **config)


def build_model(config: GPTConfig) -> GPTModel:
    model = GPTModel(config, name="small_gpt")
    model(tf.zeros((1, 2), dtype=tf.int32))
    return model

