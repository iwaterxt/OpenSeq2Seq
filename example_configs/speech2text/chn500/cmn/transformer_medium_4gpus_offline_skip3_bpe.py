# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models import Speech2Text
from open_seq2seq.encoders import TransformerEncoder
from open_seq2seq.decoders import FullyConnectedCTCDecoder
from open_seq2seq.data import Speech2TextDataLayer
from open_seq2seq.losses import CTCLoss
from open_seq2seq.optimizers.lr_policies import transformer_policy

base_model = Speech2Text
d_model = 512
num_layers = 6


base_params = {
  "random_seed": 0,
  "use_horovod": False,
  "num_gpus": 2,
  "batch_size_per_gpu": 40,

  "num_epochs": 50,
  "src_vocab_size": 0, 
  "remove_padding": False,
  "save_summaries_steps": 1000,
  "print_loss_steps": 10,
  "print_samples_steps": 10000,
  "eval_steps": 10000,
  "save_checkpoint_steps": 1000,
  "logdir": "experiments/chn500/transformer_offline_skip3_bpe_cmn",

  "optimizer": tf.contrib.opt.LazyAdamOptimizer,
  "optimizer_params": {
    "beta1": 0.9,
    "beta2": 0.997,
    "epsilon": 1e-09,
  },

  "lr_policy": transformer_policy,
  "lr_policy_params": {
    "learning_rate": 2.0,
    "warmup_steps": 8000,
    "d_model": d_model,
  },

  # weight decay
  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    'scale': 0.0005
  },

  "initializer": tf.contrib.layers.xavier_initializer,

  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],

  "encoder": TransformerEncoder,
  "encoder_params": {
    "feat_layers":
    {
        "context": [0],
        "skip_frames": 1,
        "layer_norm": False,
    },
    "encoder_layers": num_layers,
    "hidden_size": d_model,
    "num_heads": 8,
    "attention_dropout": 0.1,
    "filter_size": 4 * d_model,
    "relu_dropout": 0.1,
    "layer_postprocess_dropout": 0.1,
    "pad_embeddings_2_eight": True,
    "remove_padding": True,
    "inner_skip_params":
    {
      "inner_skip_frames": 1,
      "skip_layer": 4,
    },

  },

  "decoder": FullyConnectedCTCDecoder,
  "decoder_params": {
    "use_language_model": False,
    "infer_logits_to_pickle": True,
    # params for decoding the sequence with language model
    "beam_width": 512,
    "alpha": 2.0,
    "beta": 1.0,
    "tgt_vocab_size": 7531,
    "decoder_library_path": "ctc_decoder_with_lm/libctc_decoder_with_kenlm.so",
    "lm_path": "language_model/4-gram.binary",
    "trie_path": "language_model/trie.binary",
    "alphabet_config_path": "data/baseline_chn_2000/dict/vocab_bpe.txt",
  },
  "loss": CTCLoss,
  "loss_params": {},
}

train_params = {
  "data_layer": Speech2TextDataLayer,
  "data_layer_params": {
    "feat_format": "feat",
    "num_audio_features": 120,
    "input_type": "logfbank",
    "chn": True,
    "cache_format": "kaldi",
    "augmentation": {'time_stretch_ratio': 0.05,
                     'noise_level_min': -90,
                     'noise_level_max': -60},
    "vocab_file": "data/baseline_chn_2000/dict/vocab_bpe.txt",
    "dataset_files": [
      "data/baseline_chn_500/train_cmn/librivox-train.csv"
    ],
    "shuffle": False,
  },
}

eval_params = {
  "data_layer": Speech2TextDataLayer,
  "data_layer_params": {
    "feat_format": "feat",
    "num_audio_features": 120,
    "input_type": "logfbank",
    "chn": True,
    "cache_format": "kaldi",
    "vocab_file": "data/baseline_chn_2000/dict/vocab_bpe.txt",
    "dataset_files": [
      "data/baseline_chn_500/dev_cmn/librivox-dev.csv"
    ],
    "shuffle": False,
  },
}

infer_params = {
  "data_layer": Speech2TextDataLayer,
  "data_layer_params": {
    "num_audio_features": 120,
    "feat_format": "feat",
    "chn": True,
    "cache_format": "kaldi",
    "input_type": "logfbank",
    "vocab_file": "data/baseline_chn_2000/dict/vocab_bpe.txt",
    "dataset_files": [
      "data/baseline_chn_500/tests_cmn/ailab_tmp_asr_rand_0725_fix/librivox-ailab_tmp_asr_rand_0725_fix.csv",
    ],
    "shuffle": False,
  },
}
