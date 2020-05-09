# This code is heavily based on the code from MLPerf
# https://github.com/mlperf/reference/tree/master/translation/tensorflow
# /transformer
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf
from six.moves import range
from open_seq2seq.parts.transformer import utils
from open_seq2seq.encoders import Encoder
from open_seq2seq.encoders.ds2_encoder import splice, subsample, layer_normalize


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        # beta = tf.get_variable("bata", initializer=tf.zeros(params_shape))
        # gamma = tf.get_variable("gamma", initializer=tf.ones(params_shape))
        beta = tf.get_variable("bata", shape=params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", shape=params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            # tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs, reuse=reuse)  # (N, T_q, C)

    return outputs

def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs, reuse=reuse, scope="feed_ln")

    return outputs



class TransformerEncoderAsr(Encoder):
  """Transformer model encoder"""

  @staticmethod
  def get_required_params():
    """Static method with description of required parameters.

      Returns:
        dict:
            Dictionary containing all the parameters that **have to** be
            included into the ``params`` parameter of the
            class :meth:`__init__` method.
    """
    return dict(Encoder.get_required_params(), **{
        "encoder_layers": int,
        'feat_layers': dict,
        "hidden_size": int,
        "num_heads": int,
        "attention_dropout": float,
        "filter_size": int,
        "src_vocab_size": int,
        "relu_dropout": float,
        "layer_postprocess_dropout": float,
        "remove_padding": bool,
    })

  @staticmethod
  def get_optional_params():
    """Static method with description of optional parameters.

      Returns:
        dict:
            Dictionary containing all the parameters that **can** be
            included into the ``params`` parameter of the
            class :meth:`__init__` method.
    """
    return dict(Encoder.get_optional_params(), **{
        'regularizer': None,  # any valid TensorFlow regularizer
        'regularizer_params': dict,
        'initializer': None,  # any valid TensorFlow initializer
        'initializer_params': dict,
        'pad_embeddings_2_eight': bool,
        'norm_params': dict,
        'inner_skip_params': dict,
        'task': None,
    })

  def __init__(self, params, model, name="transformer_encoder_asr", mode='train' ):
    super(TransformerEncoderAsr, self).__init__(
        params, model, name=name, mode=mode,
    )
    self.layers = []
    self.output_normalization = None
    self._mode = mode

    self.embedding_softmax_layer = None
    self.norm_params = self.params.get("norm_params", {"type": "layernorm_L2"})
    self.regularizer = self.params.get("regularizer", None)
    if self.regularizer != None:
      self.regularizer_params = params.get("regularizer_params", {'scale': 0.0})
      self.regularizer=self.regularizer(self.regularizer_params['scale']) \
        if self.regularizer_params['scale'] > 0.0 else None


  def _encode(self, input_dict):
    training = (self.mode == "train")

    # actual encoder part
    with tf.name_scope("encode"):
      inputs, src_length = input_dict['source_tensors']
      #-------------feature layer------------------------------------
      feat_layers = self.params['feat_layers']
      context = feat_layers['context']
      skip_frames = feat_layers['skip_frames']
      layer_norm = feat_layers['layer_norm']
      if layer_norm:
        inputs = layer_normalize(
                            input_layer = inputs,
                            name = "layer_norm")

      if len(context) > 0:
        inputs = splice(
                    input_layer = inputs,
                    context = context,
                    skip_frames = skip_frames,
                    name = "splice")

      src_length = (src_length + skip_frames - 1) // skip_frames

      inner_skip_params = self.params['inner_skip_params']
      inner_skip_frames = inner_skip_params['inner_skip_frames']
      skip_layer        = inner_skip_params['skip_layer']

      # Prepare inputs to the layer stack by adding positional encodings and
      # applying dropout.

      embedded_inputs = tf.layers.dense(
                              inputs=inputs,
                              units=self.params["hidden_size"],
                              kernel_regularizer=self.regularizer,
                              activation=None,
                              name='fully_connected',
                        )

      with tf.name_scope("add_pos_encoding"):
        length = tf.shape(embedded_inputs)[1]
        pos_encoding = utils.get_position_encoding(
            length, self.params["hidden_size"],
        )
        encoder_inputs = embedded_inputs + tf.cast(x=pos_encoding,
                                                   dtype=embedded_inputs.dtype)

      if self.mode == "train":
        encoder_inputs = tf.nn.dropout(encoder_inputs,
            keep_prob = 1.0 - self.params["layer_postprocess_dropout"],
        )

      # Blocks
      for i in range(self.params['encoder_layers']):
        with tf.variable_scope("num_blocks_{}".format(i)):
          # Multihead Attention
          encoder_inputs = multihead_attention(queries=encoder_inputs,
                                               keys=encoder_inputs,
                                               num_units=self.params["hidden_size"],
                                               num_heads=self.params["num_heads"],
                                               dropout_rate=self.params["attention_dropout"],
                                               is_training=training,
                                               reuse=False,
                                               causality=False)

          # Feed Forward
          encoder_inputs = feedforward(encoder_inputs, num_units=[4 * self.params["hidden_size"], self.params["hidden_size"]],reuse=False)

      if inner_skip_frames > 1:
        src_length = (src_length + inner_skip_frames - 1) // inner_skip_frames

      return {
              'outputs': encoder_inputs,
              'src_length': src_length
      }
