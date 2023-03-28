"""
Torch model files.
"""
import torch
import typing
import numpy as np
import math

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer(
  "embedding_size",
  None,
  "Dimension of projection for embeddings."
)

flags.DEFINE_integer(
  "sequence_length",
  None,
  "Maximum sequence length allowed by Transformer-based model."
)

flags.DEFINE_float(
  "dropout_prob",
  None,
  "Set probability for dropout."
)

flags.DEFINE_integer(
  "num_attention_heads",
  None,
  "Number of heads per transformer."
)

flags.DEFINE_float(
  "layer_norm_eps",
  None,
  "Layer norm EPS"
)

flags.DEFINE_integer(
  "transformer_feedforward",
  None,
  "Feature size of transformer's FC."
)

flags.DEFINE_integer(
  "num_transformer_layers",
  None,
  "Set number of Transformer layers"
)

class PositionalEncoding(torch.nn.Module):
  def __init__(self, sequence_length: int, embedding_size: int, dropout_prob: float):
    super().__init__()
    position = torch.arange(sequence_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size))
    pe = torch.zeros(sequence_length, 1, embedding_size)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)
    self.dropout = torch.nn.Dropout(dropout_prob)
    return

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.pe[:x.size(0)]
    return self.dropout(x)

class CNN4Go(torch.nn.Module):
  def __init__(self):
    return

class Transformer4Go(torch.nn.Module):
  def __init__(self,
               vocab_size: int,
               embedding_size: int,
               padding_idx: int,
               sequence_length: int,
               dropout_prob: float,
               num_attention_heads: int,
               layer_norm_eps: float,
               transformer_feedforward: int,
               num_transformer_layers: int,
               ):
    super().__init__()
    self.embedding = torch.nn.Embedding(
      num_embeddings = vocab_size,
      embedding_dim = embedding_size,
      padding_idx = padding_idx,
    )
    self.positional_embedding = PositionalEncoding(
      sequence_length,
      embedding_size,
      dropout_prob,
    )
    encoder_layers = torch.nn.TransformerEncoderLayer(
      d_model = embedding_size,
      nhead = num_attention_heads,
      dim_feedforward = transformer_feedforward,
      dropout = dropout_prob,
      batch_first = True,
    )
    encoder_norm = torch.nn.LayerNorm(
      embedding_size,
      eps = layer_norm_eps,
    )
    self.encoder_transformer = torch.nn.TransformerEncoder(
      encoder_layer = encoder_layers,
      num_layers = num_transformer_layers,
      norm = encoder_norm,
    )
    self.mapper    = torch.nn.Linear(embedding_size, vocab_size)
    self.reducer   = torch.nn.Linear(vocab_size, 1)
    self.transpose = lambda t: torch.reshape(t, (-1, 1, sequence_length)).squeeze(1)
    self.head      = torch.nn.Linear(sequence_length, 2)
    self.embedding_size = embedding_size
    self.init_weights()
    return

  def forward(self, inputs: typing.Dict[str, torch.Tensor]) -> typing.Dict[str, torch.Tensor]:
    embedded = self.embedding(inputs)
    pos_embed = self.positional_embedding(embedded)
    hidden_state = self.encoder_transformer(
      pos_embed,
      # mask = ?,
      # src_key_padding_mask = ?
    )
    mapped_state = self.mapper(hidden_state)
    reduced_state = self.reducer(mapped_state)
    reshaped_state = self.transpose(reduced_state)
    output = self.head(reshaped_state)
    return {
      'output_logits': output
    }

  def init_weights(self) -> None:
    initrange = 0.1
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.mapper.bias.data.zero_()
    self.mapper.weight.data.uniform_(-initrange, initrange)
    self.reducer.bias.data.zero_()
    self.reducer.weight.data.uniform_(-initrange, initrange)
    self.head.weight.data.uniform_(-initrange, initrange)
    self.head.bias.data.zero_()
    return
