import numpy as np
import json
import random
import gzip
import math

class Tensorflow_data:
	def __init__(self, data_path, input_train_dir, set_name, DF_sampling=False):
		#get vocabulary information
		self.words = []
		with gzip.open(data_path + 'vocab.txt.gz', 'r') as fin:
			for line in fin:
				self.words.append(line.strip())
		self.vocab_size = len(self.words)

		#get doc sets
		print('DF vocab sampling' if DF_sampling else 'CF vocab sampling')
		self.word_count = 0
		self.vocab_distribute = np.zeros(self.vocab_size) 
		self.doc_info = []
		self.doc_text = []
		tmp = 0
		with gzip.open(input_train_dir + set_name + '.txt.gz', 'r') as fin:
			for line in fin:
				arr = line.strip().split('\t')
				self.doc_info.append((arr[0], int(arr[1]))) # (doc_id, doc_length)
				self.doc_text.append([int(i) for i in arr[2].split(' ')])
				if DF_sampling:
					for idx in set(self.doc_text[-1]):
						self.vocab_distribute[idx] += 1	
				else:
					for idx in self.doc_text[-1]:
						self.vocab_distribute[idx] += 1
				self.word_count += len(self.doc_text[-1])

		self.doc_num = len(self.doc_info)
		self.vocab_distribute = self.vocab_distribute.tolist() 
		self.sub_sampling_rate = None

		print("Data statistic: vocab %d, doc %d\n" % (self.vocab_size, self.doc_num))

	def sub_sampling(self, subsample_threshold):
		if subsample_threshold == 0.0:
			return
		self.sub_sampling_rate = np.ones(self.vocab_size)
		threshold = sum(self.vocab_distribute) * subsample_threshold
		count_sub_sample = 0
		for i in xrange(self.vocab_size):
			#vocab_distribute[i] could be zero if the word does not appear in the training set
			self.sub_sampling_rate[i] = min((np.sqrt(float(self.vocab_distribute[i]) / threshold) + 1) * threshold / float(self.vocab_distribute[i]),
											1.0)
			count_sub_sample += 1

	def output_embedding(self, embeddings, emb_info_list, output_file_name):
		with gzip.open(output_file_name + '.gz', 'wb') as emb_fout:
			try:
				length = len(emb_info_list)
				if length < 1:
					return
				dimensions = len(embeddings[0])
				emb_fout.write(str(length) + '\n')
				emb_fout.write(str(dimensions) + '\n')
				for i in xrange(length):
					emb_fout.write(emb_info_list[i] + '\t')
					for j in xrange(dimensions):
						emb_fout.write(str(embeddings[i][j]) + ' ')
					emb_fout.write('\n')
			except:
				emb_fout.write(str(embeddings) + ' ')





