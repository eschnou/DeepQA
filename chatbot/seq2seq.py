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
                       thought_map=True,
                       dtype=None):
  """
  Experimenting with the concept of thought mapping
  """
  # First we define the encoder part
  with tf.variable_scope("thought_encoder") as scope:
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

  # Insert a thought map between encoder and decoder
  decoder_initial_state = thought_mapper(encoder_state) if thought_map else encoder_state

  # Build the decoder
  with tf.variable_scope("thought_decoder") as scope:

    if output_projection is not None:
      dtype = scope.dtype
      proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
      proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
      proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
      proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    embedding = tf.get_variable("embedding", [num_decoder_symbols, embedding_size])

    loop_function = tf.nn.seq2seq._extract_argmax_and_embed(
        embedding, output_projection,
        update_embedding=False) if feed_previous else None

    emb_inp = (
        tf.nn.embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs)

    return tf.nn.seq2seq.rnn_decoder(emb_inp, decoder_initial_state, cell, loop_function=loop_function)

def thought_mapper(encoder_state, thoughtmap_layers=2, thoughtmap_layer_size=4096):
      """The following is an intermediate layer whose role is to map a question 'thought'
         to an answer thought. In the future, this layer can also use additional inputs
         for example to take into account the context of the question, or to access something
         like a memory.
      """

      with tf.variable_scope("thought_mapper", initializer=xavier_weight_init()) as scope:

          thought_vector = []
          decoder_initial_state = []
          num_layers = len(encoder_state)
          hidden_size = encoder_state[0].get_shape()[1].value
          thought_size = num_layers * hidden_size

          # Collect all internal states vector and flatten a single thought vector t with size numLayers * hiddenSize
          for i in range (len(encoder_state)):
              state = encoder_state[i]
              thought_vector.append(state)
          thought_vector = tf.concat(1, thought_vector)

          # A first layer to map between the encoded thought vector to the thought mapper
          TM_W_in = tf.get_variable("TM_W_in", [thought_size, thoughtmap_layer_size])
          TM_b_in = tf.get_variable("TM_b_in", [thoughtmap_layer_size])
          layer = tf.map_fn(lambda x: tf.squeeze(tf.nn.relu(tf.matmul(tf.reshape(x, [-1, thought_size]), TM_W_in) + TM_b_in)), thought_vector)

          # Add hidden layers in thought mapper
          for i in range(thoughtmap_layers):
              W = tf.get_variable("TM_W_%d" % i, [thoughtmap_layer_size, thoughtmap_layer_size])
              b = tf.get_variable("TM_b_%d" % i, [thoughtmap_layer_size])
              layer = tf.map_fn(lambda x: tf.squeeze(tf.nn.relu(tf.matmul(tf.reshape(x, [-1, thoughtmap_layer_size]), W) + b)), layer)

          # Add the output layer feeding the encoded thought to the decoder
          TM_W_out = tf.get_variable("TM_W_out", [thoughtmap_layer_size, thought_size])
          TM_b_out = tf.get_variable("TM_b_out", [thought_size])
          layer = tf.map_fn(lambda x: tf.squeeze(tf.nn.relu(tf.matmul(tf.reshape(x, [-1, thoughtmap_layer_size]), TM_W_out) + TM_b_out)), layer)

          # Re-split the question thought vector to create the decoder state
          layers = tf.split(1, num_layers, layer)

          for l in layers:
              decoder_initial_state.append(l)

          return decoder_initial_state


def xavier_weight_init():
   """
   Returns function that creates random tensor.

   The specified function will take in a shape (tuple or 1-d array) and must
   return a random tensor of the specified shape and must be drawn from the
   Xavier initialization distribution.
   """
   def _xavier_initializer(shape, **kwargs):
     """Defines an initializer for the Xavier distribution.

     This function will be used as a variable scope initializer.
     https://www.tensorflow.org/versions/r0.7/how_tos/variable_scope/index.html#initializers-in-variable-scope

     Args:
       shape: Tuple or 1-d array that species dimensions of requested tensor.
     Returns:
       out: tf.Tensor of specified shape sampled from Xavier distribution.
     """

     m = shape[0]
     n = shape[1] if len(shape) > 1 else shape[0]

     bound = np.sqrt(6) / np.sqrt(m + n)
     out = tf.random_uniform(shape, minval=-bound, maxval=bound)

     return out
   # Returns defined initializer function.
   return _xavier_initializer
