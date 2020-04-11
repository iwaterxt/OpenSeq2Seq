
import kaldi_io
import pickle
import numpy as np
import sys


def load_dump(pickle_file):
  with open(pickle_file, 'rb') as f:
    data = pickle.load(f, encoding='bytes')
  return data

def get_logits(data):

  logits = data['logits']
  return logits

def load_map(feats_scp):
	feat2utt = {}
	with open(feats_scp, 'rb') as f:
		lines = f.readlines()
		for line in lines:
			l = line.strip().split()
			encoding = 'utf-8'
			feat2utt[str(l[1], encoding)] = str(l[0], encoding)
	return feat2utt

def softmax(x):
  m = np.expand_dims(np.max(x, axis=-1), -1)
  e = np.exp(x - m)
  return e / np.expand_dims(e.sum(axis=-1), -1)

def main(pickle_file, ark_file, feats_scp):
	data = load_dump(pickle_file)
	poster_dict = get_logits(data)
	feat2utt = load_map(feats_scp)
	with open(ark_file,'wb') as f:
		for key,mat in poster_dict.items():
			r,l = mat.shape
			mat_new = np.zeros((r,l))
			mat_new[:,0] = mat[:, l-1]
			mat_new[:,1:l-1] = mat[:, 0:l-2]
			kaldi_io.write_mat(f, np.log(softmax(mat_new)), key=feat2utt[key])


if __name__ == "__main__":
	if len(sys.argv) != 4:
		print ("python3 decode_poster_prepare.py pickle_file ark_file feats_scp")
		sys.exit(1)
	main(sys.argv[1], sys.argv[2], sys.argv[3])
