# This code is heavily based on the code from MLPerf
# https://github.com/mlperf/reference/tree/master/translation/tensorflow
# /transformer
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf
from six.moves import range

from open_seq2seq.encoders import Encoder
from open_seq2seq.encoders.ds2_encoder import splice, subsample, layer_normalize
from open_seq2seq.parts.transformer import attention_layer, ffn_layer, utils, \
                                           embedding_layer
from open_seq2seq.parts.transformer.common import PrePostProcessingWrapper, \
                                    LayerNormalization, Transformer_BatchNorm

from open_seq2seq.parts.cnns.conv_blocks import conv_bn_actv

class TransformerEncoder(Encoder):
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
        "feat_layers": dict,
        "conv_layers": list,
        "activation_fn": None,  # any valid callable
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
        'data_format': ['channels_first', 'channels_last', 'BCTF', 'BTFC', 'BCFT', 'BFTC'],
        'bn_momentum': float,
        'bn_epsilon': float,
    })

  def __init__(self, params, model, name="transformer_encoder", mode='train' ):
    super(TransformerEncoder, self).__init__(
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


  def _call(self, encoder_inputs, attention_bias, inputs_padding, inner_skip_frames=1, skip_layer=1):
    for n, layer in enumerate(self.layers):
      # Run inputs through the sublayers.
      self_attention_layer = layer[0]
      feed_forward_network = layer[1]

      with tf.variable_scope("layer_%d" % n):
        with tf.variable_scope("self_attention"):
          encoder_inputs = self_attention_layer(encoder_inputs, attention_bias)
        with tf.variable_scope("ffn"):
          encoder_inputs = feed_forward_network(encoder_inputs, inputs_padding)
      if inner_skip_frames > 1 and n == skip_layer:
        encoder_inputs = subsample(
                        input_layer = encoder_inputs,
                        regularizer = self.regularizer,
                        skip_frames = inner_skip_frames,
                        name = "subsample_inner")

    if self.params["task"] == "ASR":
        return encoder_inputs
    else:
        return self.output_normalization(encoder_inputs)

  def _encode(self, input_dict):
    training = (self.mode == "train")
    data_format = self.params.get('data_format', 'channels_last')
    bn_momentum = self.params.get('bn_momentum', 0.99)
    bn_epsilon = self.params.get('bn_epsilon', 1e-3)
    if len(self.layers) == 0:
      # prepare encoder graph
      self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
          self.params["src_vocab_size"], self.params["hidden_size"],
          pad_vocab_to_eight=self.params.get('pad_embeddings_2_eight', False),
      )

      for _ in range(self.params['encoder_layers']):
        # Create sublayers for each layer.
        self_attention_layer = attention_layer.SelfAttention(
          hidden_size=self.params["hidden_size"],
          num_heads=self.params["num_heads"],
          attention_dropout=self.params["attention_dropout"],
          train=training,
          regularizer=self.regularizer
        )

        feed_forward_network = ffn_layer.FeedFowardNetwork(
          hidden_size=self.params["hidden_size"],
          filter_size=self.params["filter_size"],
          relu_dropout=self.params["relu_dropout"],
          train=training,
          regularizer=self.regularizer
        )

        self.layers.append([
            PrePostProcessingWrapper(self_attention_layer, self.params,
                                     training),
            PrePostProcessingWrapper(feed_forward_network, self.params,
                                     training)
        ])

      # final normalization layer.
      print("Encoder:", self.norm_params["type"], self.mode)
      if self.norm_params["type"] =="batch_norm":
        self.output_normalization = Transformer_BatchNorm(
          training=training,
          params=self.norm_params)
      else:
        self.output_normalization = LayerNormalization(
          hidden_size=self.params["hidden_size"], params=self.norm_params)

    # actual encoder part
    with tf.name_scope("encode"):
      feats, src_lengths = input_dict['source_tensors']
      #-------------feature layer------------------------------------
      feat_layers = self.params['feat_layers']
      context = feat_layers['context']
      skip_frames = feat_layers['skip_frames']
      layer_norm = feat_layers['layer_norm']
      if layer_norm:
        feats = layer_normalize(
                            input_layer = feats,
                            name = "layer_norm")

      if len(context) > 0:
        feats = splice(
                    input_layer = feats,
                    context = context,
                    skip_frames = skip_frames,
                    name = "splice")

      src_lengths = (src_lengths + skip_frames - 1) // skip_frames

      inner_skip_params = self.params['inner_skip_params']
      inner_skip_frames = inner_skip_params['inner_skip_frames']
      skip_layer        = inner_skip_params['skip_layer']

      input_layer = tf.expand_dims(feats, axis=-1) # BTFC

      batch_size = input_layer.get_shape().as_list()[0] #number of streams
      freq = input_layer.get_shape().as_list()[2] #feature dim

      # supported data_formats:
      #    BTFC = channel_last (legacy)
      #    BCTF = channel_first(legacy)
      #    BFTC
      #    BCFT

      if data_format=='channels_last' or data_format=='BTFC':
        layout  = 'BTFC'
        dformat = 'channels_last'
      elif data_format=='channels_first' or data_format=='BCTF':
        layout  = 'BCTF'
        dformat = 'channels_first'
      elif data_format=='BFTC':
        layout  = 'BFTC'
        dformat = 'channels_last'
      elif data_format=='BCFT':
        layout  = 'BCFT'
        dformat = 'channels_first'
      else:
        print("WARNING: unsupported data format: will use channels_last (BTFC) instead")
        layout  = 'BTFC'
        dformat = 'channels_last'

      #input_layer is BTFC

      if   layout == 'BCTF':
        top_layer = tf.transpose(input_layer, [0, 3, 1, 2])
      elif layout == 'BFTC':
        top_layer = tf.transpose(input_layer, [0, 2, 1, 3])
      elif layout == 'BCFT':
        top_layer = tf.transpose(input_layer, [0, 3, 2, 1])
      else:
        top_layer = input_layer

      # print("<<< pre-conv:", top_layer.get_shape().as_list())

      # ----- Convolutional layers ---------------------------------------------
      conv_layers = self.params['conv_layers']

      for idx_conv in range(len(conv_layers)):
        ch_out = conv_layers[idx_conv]['num_channels']
        kernel_size = conv_layers[idx_conv]['kernel_size']  # [T,F] format
        strides = conv_layers[idx_conv]['stride']           # [T,F] format
        padding = conv_layers[idx_conv]['padding']

        if padding == "VALID":
          src_lengths = (src_lengths - kernel_size[0] + strides[0]) // strides[0]
          freq = (freq - kernel_size[1] + strides[1]) // strides[1]
        else:
          src_lengths = (src_lengths + strides[0] - 1) // strides[0]
          freq = (freq + strides[1] -1) // strides[1]

        if layout == 'BFTC' or layout == 'BCFT':
          kernel_size = kernel_size[::-1]
          strides = strides[::-1]

        top_layer = conv_bn_actv(
            layer_type="conv2d",
            name="conv{}".format(idx_conv + 1),
            inputs=top_layer,
            filters=ch_out,
            kernel_size=kernel_size,
            activation_fn=self.params['activation_fn'],
            strides=strides,
            padding=padding,
            regularizer=self.regularizer,
            training=training,
            data_format=dformat,
            bn_momentum=bn_momentum,
            bn_epsilon=bn_epsilon,
        )

      # convert layout --> BTFC
      # if data_format == 'channels_first':
      #   top_layer = tf.transpose(top_layer, [0, 2, 3, 1])

      if   layout == 'BCTF': # BCTF --> BTFC
        top_layer = tf.transpose(top_layer, [0, 2, 3, 1])
      elif layout == 'BFTC': # BFTC --> BTFC
        top_layer = tf.transpose(top_layer, [0, 2, 1, 3])
      elif layout == 'BCFT': # BCFT --> BTFC
        top_layer = tf.transpose(top_layer, [0, 3, 2, 1])



      # reshape to [B, T, FxC]
      f = top_layer.get_shape().as_list()[2]
      c = top_layer.get_shape().as_list()[3]
      fc = f * c
      inputs = tf.reshape(top_layer, [batch_size, -1, fc])
      # Prepare inputs to the layer stack by adding positional encodings and
      # applying dropout.
      if self.params["src_vocab_size"] > 0:
        embedded_inputs = self.embedding_softmax_layer(inputs)
      else:
        embedded_inputs = tf.layers.dense(
                              inputs=inputs,
                              units=self.params["hidden_size"],
                              kernel_regularizer=self.regularizer,
                              activation=None,
                              name='fully_connected',
                          )
      # Padding should be pay attention
      if self.params["task"] == "ASR":
        if self.params["remove_padding"]:
            inputs_padding = utils.get_padding(tf.squeeze(feats[:,:,0]))
        else:
            inputs_padding = None
        inputs_attention_bias = utils.get_padding_bias(tf.squeeze(feats[:,:,0]))
      else:
        if self.params["remove_padding"]:
            inputs_padding = utils.get_padding(embedded_inputs)
        else:
            inputs_padding = None
        inputs_attention_bias = utils.get_padding_bias(embedded_inputs)

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

      encoded = self._call(encoder_inputs, inputs_attention_bias,
                           inputs_padding, inner_skip_frames, skip_layer)
      if inner_skip_frames > 1:
        src_lengths = (src_lengths + inner_skip_frames - 1) // inner_skip_frames

      return {'outputs': encoded,
              'inputs_attention_bias': inputs_attention_bias,
              'state': None,
              'src_length': src_lengths,
              'embedding_softmax_layer': self.embedding_softmax_layer,
              'encoder_input': inputs}
