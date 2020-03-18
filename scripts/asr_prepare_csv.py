#!/usr/bin/env python

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2018 Mozilla Corporation


from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import codecs
import fnmatch
import pandas
import tqdm
import subprocess
import tarfile
import unicodedata

from sox import Transformer
import urllib
from tensorflow.python.platform import gfile




def preprocess_data(data_dir):
	print("Converting WAV and transcriptions...")
	with tqdm.tqdm(total=2) as bar:
		train = _convert_audio_and_sentences(data_dir, 'train')
		bar.update(1)
		dev = _convert_audio_and_sentences(data_dir, 'dev')
		bar.update(1)
	# Write sets to disk as CSV files
	train_dir = os.path.join(data_dir, 'train')
  	train.to_csv(os.path.join(train_dir, "librivox-train.csv"), index=False)
  	dev_dir = os.path.join(data_dir, 'dev')
  	dev.to_csv(os.path.join(dev_dir, "librivox-dev.csv"), index=False)


def _convert_audio_and_sentences(source_dir):

  files = []
  utt2trans = {}

  wav_scp = os.path.join(source_dir, "wav.scp")
  if not os.path.exists(wav_scp):
  	print (" not exists wav.scp file!")
  	sys.exit(1)

  text_file = os.path.join(source_dir, "text")
  if not os.path.exists(text_file)
  	print ("not exists text file!")
  	sys.exit(1)

  with codecs.open(text_file, "r", "utf-8") as fin:
  	for line in fin:
  		line = line.split().strip()
  		utt2trans[line[0]] = " ".join(line[1:])

  with codecs.open(wav_scp, "r", "utf-8") as fin:
  	for line in fin:
  		line = line.split().strip()
  		wav_filesize = os.path.getsize(str(line[1]))
  		files.append(str(line[1]), wav_filesize, utt2trans[line[0]])

  return pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "transcript"])


def _convert_feat_and_sentences(source_dir):

  files = []
  utt2trans = {}
  feats_scp = os.path.join(source_dir, "feats.scp")

  if not os.path.exists(feats_scp):
  	print (" not exists feats.scp file!")
  	sys.exit(1)


  text_file = os.path.join(source_dir, "text")
  if not os.path.exists(text_file)
  	print ("not exists text file !")
  	sys.exit(1)
  with codecs.open(text_file, "r", "utf-8") as fin:
  	for line in fin:
  		line = line.split().strip()
  		utt2trans[line[0]] = " ".join(line[1:])

  with codecs.open(feats_scp, "r", "utf-8") as fin:
  	for line in fin:
  		line = line.split().strip()
  		feats_filesize = os.path.getsize(str(line[1]))
  		files.append(str(line[1]), feats_filesize, utt2trans[line[0]])

  return pandas.DataFrame(data=files, columns=["feats_filename", "feats_filesize", "transcript"])


if __name__ == "__main__":
  preprocess_data(sys.argv[1])