# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np

def custom_rnn_seq2seq(encoder_inputs,
                       decoder_inputs,
                       cell,
                       num_encoder_symbols,
                       num_decoder_symbols,
                       embedding_size,
                       output_projection=None,
                       feed_previous=False,
                       dtype=None,):
  """Embedding RNN sequence-to-sequence model.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_encoder_symbols x input_size]). Then it runs an RNN to encode
  embedded encoder_inputs into a state vector. Next, it embeds decoder_inputs
  by another newly created embedding (of shape [num_decoder_symbols x
  input_size]). Then it runs RNN decoder, initialized with the last
  encoder state, on embedded decoder_inputs.

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols: Integer; number of symbols on the decoder side.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_decoder_symbols] and B has
      shape [num_decoder_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial state for both the encoder and encoder
      rnn cells (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_rnn_seq2seq"

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors. The
        output is of shape [batch_size x cell.output_size] when
        output_projection is not None (and represents the dense representation
        of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
        when output_projection is None.
      state: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with tf.variable_scope("custom_rnn_seq2seq") as scope:
    if dtype is not None:
      scope.set_dtype(dtype)
    else:
      dtype = scope.dtype

    encoder_cell = tf.nn.rnn_cell.EmbeddingWrapper(
        cell, embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size)

    # encoder_state is a list of state for each hidden layer
    _, encoder_state = tf.nn.rnn(encoder_cell, encoder_inputs, dtype=dtype)

    if output_projection is None:
      cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)

  # The following is an intermediate layer whose role is to map a question 'thought'
  # to an answer thought. In the future, this layer can also use additional inputs
  # for example to take into account the context of the question, or to access something
  # like a memory.
  with tf.variable_scope("thought_mapper") as scope:
      decoder_state = []
      hidden_size = encoder_state[0].c.get_shape()[1].value
      batch_size = len(encoder_inputs)

      for i in range (len(encoder_state)):
          state = encoder_state[i]
          Wc = tf.get_variable("Wc%d" % i, initializer = tf.convert_to_tensor(np.eye(hidden_size), dtype=tf.float32), trainable=False)
          bc = tf.get_variable("bc%d" %i,  initializer = tf.zeros([hidden_size]), dtype=tf.float32, trainable=False)
          Wh = tf.get_variable("Wh%d" % i, initializer = tf.convert_to_tensor(np.eye(hidden_size), dtype=tf.float32), trainable=False)
          bh = tf.get_variable("bh%d" %i,  initializer = tf.zeros([hidden_size]),  dtype=tf.float32, trainable=False)
          c = tf.map_fn(lambda x: tf.squeeze(tf.matmul(tf.reshape(x, [-1, hidden_size]), Wc) + bc), state.c)
          h = tf.map_fn(lambda x: tf.squeeze(tf.matmul(tf.reshape(x, [-1, hidden_size]), Wh) + bh), state.h)
          decoder_state.append(tf.nn.rnn_cell.LSTMStateTuple(c, h))

  with tf.variable_scope("embedding_rnn_decoder") as scope:

    if output_projection is not None:
      dtype = scope.dtype
      proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
      proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
      proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
      proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    embedding = tf.get_variable("embedding", [num_decoder_symbols, embedding_size])

    loop_function = tf.nn.seq2seq_extract_argmax_and_embed(
        embedding, output_projection,
        update_embedding_for_previous) if feed_previous else None

    emb_inp = (
        tf.nn.embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs)

    return tf.nn.seq2seq.rnn_decoder(emb_inp, decoder_state, cell, loop_function=loop_function)
