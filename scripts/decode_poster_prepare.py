
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

def softmax(x):
  m = np.expand_dims(np.max(x, axis=-1), -1)
  e = np.exp(x - m)
  return e / np.expand_dims(e.sum(axis=-1), -1)

def main(pickle_file, ark_file):
	data = load_dump(pickle_file)
	poster_dict = get_logits(data)
	with open(ark_file,'wb') as f:
		for key,mat in poster_dict.items():
			r,l = mat.shape
			end_col = np.copy(mat[:, l-1])
			mat_new = np.hstack((end_col, mat[:, 0:l-2]))
			kaldi_io.write_mat(f, np.log(softmax(mat_new)), key=key)


if __name__ == "__main__":
	if len(sys.argv) != 3:
		print ("python3 decode_poster_prepare.py pickle_file ark_file")
		sys.exit(1)
	main(sys.argv[1], sys.argv[2])
