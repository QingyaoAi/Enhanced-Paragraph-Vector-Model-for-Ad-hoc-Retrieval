from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange# pylint: disable=redefined-builtin
from six.moves import zip	 # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops import variable_scope
import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange# pylint: disable=redefined-builtin
import tensorflow as tf

# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access


class ParagraphVector(object):
	def __init__(self, vocab_size, doc_num,
				 vocab_distribute, window_size,
				 embed_size, max_gradient_norm, batch_size, learning_rate, L2_lambda,
				 net_struct, similarity_func, distort, forward_only=False, negative_samples = 5):
		"""Create the model.
	
		Args:
			vocab_size: the number of words in the corpus.
			doc_num: the number of documents in the corpus.
			embed_size: the size of each embedding
			window_size: the size of half context window
			vocab_distribute: the distribution for words, used for negative sampling
			max_gradient_norm: gradients will be clipped to maximally this norm.
			batch_size: the size of the batches used during training;
			the model construction is not independent of batch_size, so it cannot be
			changed after initialization.
			learning_rate: learning rate to start with.
			learning_rate_decay_factor: decay learning rate by this much when needed.
			distort: distort rate for vocab distribution
			forward_only: if set, we do not construct the backward pass in the model.
			negative_samples: the number of negative_samples for training
		"""
		self.vocab_size = vocab_size
		self.doc_num = doc_num
		self.negative_samples = negative_samples
		self.embed_size = embed_size
		self.window_size = window_size
		self.vocab_distribute = vocab_distribute
		self.max_gradient_norm = max_gradient_norm
		#self.batch_size = batch_size * (self.negative_samples + 1)
		self.batch_size = batch_size
		self.init_learning_rate = learning_rate
		self.distort = distort
		self.L2_lambda = L2_lambda
		self.net_struct = net_struct
		self.similarity_func = similarity_func
		self.global_step = tf.Variable(0, trainable=False)

		# Feeds for inputs.
		self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
		self.word_idxs = tf.placeholder(tf.int64, shape=[None], name="word_idxs") 
		self.doc_idxs = tf.placeholder(tf.int64, shape=[None], name="doc_idxs")
		if self.L2_lambda > 0:
			self.doc_lengths = tf.placeholder(tf.float32, shape=[None], name="doc_lengths")
		
		# setup model
		self.need_context = False
		if 'cbow' in self.net_struct or 'hdc' in self.net_struct or 'pdc' in self.net_struct:
			self.need_context = True
		self.need_bias = False
		if 'bias' in self.net_struct:
			self.need_bias = True

		if self.need_context:
			self.context_word_idxs = []
			for i in xrange(2 * self.window_size):
				self.context_word_idxs.append(tf.placeholder(tf.int64, shape=[None], name="context_idx{0}".format(i)))

		print('L2 lambda ' + str(self.L2_lambda))

		# Training losses.
		self.loss = self.build_embedding_graph_and_loss()

		# Gradients and SGD update operation for training the model.
		params = tf.trainable_variables()
		if not forward_only:
			
			if 'sgd' in self.net_struct:
				print('SGD Gradients')
				opt = tf.train.GradientDescentOptimizer(self.learning_rate)
			else:
				print('Ada Gradients')
				opt = tf.train.AdagradOptimizer(self.init_learning_rate)
			
			self.gradients = tf.gradients(self.loss, params)
			
			if self.max_gradient_norm > 0:
				self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients,
																	 self.max_gradient_norm)
				self.updates = opt.apply_gradients(zip(self.clipped_gradients, params),
											 global_step=self.global_step)
			else:
				print('No gradient clipping')
				self.updates = opt.apply_gradients(zip(self.gradients, params),
										 global_step=self.global_step)
		else:
			self.doc_softmax_norm = self.compute_softmax_norm_for_doc()
	
		self.saver = tf.train.Saver(tf.global_variables())
	
	def build_embedding_graph_and_loss(self, scope = None):
		with variable_scope.variable_scope(scope or "embedding_graph"):									
			batch_size = array_ops.shape(self.doc_idxs)[0]#get batch_size										
			# Word embeddings.									
			init_width = 0.5 / self.embed_size									
			self.word_emb = tf.Variable(tf.random_uniform(									
								[self.vocab_size, self.embed_size], -init_width, init_width),				
								name="word_emb")
			self.word_bias = None
			if self.need_bias:
				self.word_bias = tf.Variable(tf.zeros([self.vocab_size]), name="word_b")

			# doc embeddings.							
			self.doc_emb = tf.Variable( tf.zeros([self.doc_num, self.embed_size]),								
									name="doc_emb")
			#self.doc_emb = tf.Variable(tf.random_uniform(									
			#					[self.doc_num, self.embed_size], -init_width*self.embed_size, init_width*self.embed_size), 
			#					name="doc_emb")	
			
			# context embedding
			if self.need_context:
				self.context_emb = tf.Variable( tf.zeros([self.vocab_size, self.embed_size]),								
									name="context_emb")
				#self.context_emb = tf.Variable(tf.random_uniform(									
				#						[self.vocab_size, self.embed_size], -init_width, init_width),								
				#						name="context_emb")
				self.context_bias = None
				if self.need_bias:
					self.context_bias = tf.Variable(tf.zeros([self.vocab_size]), name="context_b")
										
			loss = None
			regularization_terms = []
			def aggregate_context_vectors(idxs,emb):
				embs = tf.nn.embedding_lookup(emb, idxs)
				return tf.reduce_sum(embs, 0)				
			#build model
			doc_vec = tf.nn.embedding_lookup(self.doc_emb, self.doc_idxs)
			if 'cbow' in self.net_struct:
				print('Model: PV-CBOW')
				agg_context_vector = aggregate_context_vectors(tf.stack(self.context_word_idxs), self.context_emb)
				print(agg_context_vector.get_shape())
				example_vec = agg_context_vector + doc_vec
				loss, ew_emb = self.single_nce_loss(example_vec, self.word_idxs, self.word_emb, self.word_bias,											
					self.vocab_size, self.negative_samples, self.vocab_distribute, self.distort)
				regularization_terms = ew_emb
			else:
				print('Model: PV-DBOW')
				loss, dw_emb = self.single_nce_loss(doc_vec, self.word_idxs, self.word_emb, self.word_bias,											
					self.vocab_size, self.negative_samples, self.vocab_distribute, self.distort)
				regularization_terms = dw_emb
				
				if 'pdc' in self.net_struct:
					print('PDC')
					agg_context_vector = aggregate_context_vectors(tf.stack(self.context_word_idxs), self.context_emb)
					pdc_loss, pdc_emb = self.single_nce_loss(agg_context_vector, self.word_idxs, self.word_emb, self.word_bias,
						self.vocab_size, self.negative_samples, self.vocab_distribute, self.distort)
					loss += pdc_loss
					regularization_terms.append(pdc_emb[0])
					regularization_terms.append(pdc_emb[2])
				if 'hdc' in self.net_struct:
					print('HDC')
					word_vec = tf.nn.embedding_lookup(self.word_emb, self.word_idxs)
					for context_word_idx in self.context_word_idxs:
						c_loss, c_emb = self.single_nce_loss(word_vec, context_word_idx, self.context_emb, self.context_bias, 											
							self.vocab_size, self.negative_samples, self.vocab_distribute, self.distort)
						loss += c_loss
						regularization_terms.append(c_emb[1])
						regularization_terms.append(c_emb[2])
				
			
			# L2 regularization
			if self.L2_lambda > 0:
				print('L2 regularization')
				l2_loss = tf.nn.l2_loss(regularization_terms[0]/tf.reshape(self.doc_lengths, [-1,1]))#tf.reduce_sum(tf.norm(regularization_terms[0], axis=1)/self.doc_lengths)
				for i in xrange(1,len(regularization_terms)):
					l2_loss += tf.nn.l2_loss(regularization_terms[i])
				loss += self.L2_lambda * l2_loss

			#loss = tf.Print(loss, [loss], 'this is loss', summarize=5)
			#tmp_batch_size = math_ops.cast(batch_size, dtypes.float32)								
			#tmp_batch_size = tf.Print(tmp_batch_size, [tmp_batch_size], 'this is tmp_batch_size', summarize=5)
			#tmp_loss = loss / tmp_batch_size
			#tmp_loss = tf.Print(tmp_loss, [tmp_loss], 'this is tmp_loss', summarize=5)

			# if divided by batch size, then we need much larger learning
			return loss / math_ops.cast(batch_size, dtypes.float32)

	def compute_softmax_norm_for_doc(self):
		doc_vec = tf.nn.embedding_lookup(self.doc_emb, self.doc_idxs)
		doc_norms = tf.matmul(doc_vec, self.word_emb, transpose_b=True)
		#doc_norms = tf.Print(doc_norms, [doc_norms], 'this is doc_norms', summarize=5)
		exp_doc_norms = tf.exp(doc_norms)
		#exp_doc_norms = tf.Print(exp_doc_norms, [exp_doc_norms], 'this is exp_doc_norms', summarize=5)
		return tf.reduce_sum(exp_doc_norms, 1)


	def step(self, session, learning_rate, word_idxs, context_word_idxs, 
				doc_idxs, doc_word_idxs, doc_lengths, forward_only, test_mode = 'product_scores'):
		"""Run a step of the model feeding the given inputs.
	
		Args:
			session: tensorflow session to use.
			learning_rate: the learning rate of current step
			word_idxs: A numpy [1] float vector.
			context_word_idxs: A list of numpy [1] float vector.
			doc_idxs: A numpy [1] float vector.
			doc_word_idxs: A numpy [None, x] float vector.
			forward_only: whether to do the update step or only forward.
	
		Returns:
			A triple consisting of gradient norm (or None if we did not do backward),
			average perplexity, and the outputs.
	
		Raises:
			ValueError: if length of encoder_inputs, decoder_inputs, or
			target_weights disagrees with bucket size for the specified bucket_id.
		"""
	
		# Input feed: encoder inputs, decoder inputs, target_weights, as provided.
		input_feed = {}
		input_feed[self.learning_rate.name] = learning_rate
		input_feed[self.word_idxs.name] = word_idxs
		input_feed[self.doc_idxs.name] = doc_idxs
		if self.L2_lambda > 0:
			input_feed[self.doc_lengths.name] = doc_lengths
		
		#print(word_idxs)
		#print(doc_idxs)

		if self.need_context:
			for i in xrange(2 * self.window_size):
				input_feed[self.context_word_idxs[i].name] = context_word_idxs[i]
	
		# Output feed: depends on whether we do a backward step or not.
		if not forward_only:
			output_feed = [self.updates,	# Update Op that does SGD.
						 #self.norm,	# Gradient norm.
						 self.loss]	# Loss for this batch.
		else:
			if test_mode == 'output_embedding':
				output_feed = [self.word_emb, self.doc_emb]
				if self.need_context:
					output_feed += [self.context_emb]
			else:
				output_feed = [self.doc_softmax_norm] #negative instance output
			
		outputs = session.run(output_feed, input_feed)
		if not forward_only:
			return outputs[1], None	# loss, no outputs, Gradient norm.
		else:
			if test_mode == 'output_embedding':
				return outputs[:2], outputs[2:]
			else:
				return outputs[0], None	# product scores to input user

	def setup_data_set(self, data_set, words_to_train):
		self.data_set = data_set
		self.words_to_train = words_to_train
		self.finished_word_num = 0
		if 'cbow' in self.net_struct or 'hdc' in self.net_struct:
			self.need_context = True

	def intialize_epoch(self, training_seq):
		self.train_seq = training_seq
		self.doc_num = len(self.train_seq)
		self.cur_doc_i = 0
		self.cur_word_i = 0

	def get_train_batch(self):
		doc_idxs, word_idxs, doc_lengths, context_word_idxs = [],[],[],[]
		learning_rate = self.init_learning_rate * max(0.0001, 
									1.0 - self.finished_word_num / self.words_to_train)
		doc_idx = self.train_seq[self.cur_doc_i]
		doc_length = self.data_set.doc_info[doc_idx][1]
		text_list = self.data_set.doc_text[doc_idx]
		text_length = len(text_list)
		while len(word_idxs) < self.batch_size:
			#print('doc %d word %d word_idx %d' % (doc_idx, self.cur_word_i, text_list[self.cur_word_i]))
			#if sample this word
			if self.data_set.sub_sampling_rate == None or random.random() < self.data_set.sub_sampling_rate[text_list[self.cur_word_i]]:
				doc_idxs.append(doc_idx)
				doc_lengths.append(doc_length)
				word_idxs.append(text_list[self.cur_word_i])
				if self.need_context:
					i = self.cur_word_i
					start_index = i - self.window_size if i - self.window_size > 0 else 0
					context_word_list = text_list[start_index:i] + text_list[i+1:i+self.window_size+1]
					while len(context_word_list) < 2 * self.window_size:
						context_word_list += text_list[start_index:start_index+2*self.window_size-len(context_word_list)]
					context_word_idxs.append(context_word_list)

			#move to the next
			self.cur_word_i += 1
			self.finished_word_num += 1
			if self.cur_word_i == text_length:
				self.cur_doc_i += 1
				if self.cur_doc_i == self.doc_num:
					break
				self.cur_word_i = 0
				doc_idx = self.train_seq[self.cur_doc_i]
				doc_length = self.data_set.doc_info[doc_idx][1]
				text_list = self.data_set.doc_text[doc_idx]
				text_length = len(text_list)

		batch_context_word_idxs = None
		length = len(word_idxs)
		if self.need_context:
			batch_context_word_idxs = []
			for length_idx in xrange(2 * self.window_size):
				batch_context_word_idxs.append(np.array([context_word_idxs[batch_idx][length_idx]
						for batch_idx in xrange(length)], dtype=np.int64))

		has_next = False if self.cur_doc_i == self.doc_num else True
		return word_idxs, batch_context_word_idxs, doc_idxs, None, doc_lengths, learning_rate, has_next

	
	def prepare_test_epoch(self, test_seq):
		self.test_seq = test_seq
		self.doc_num = len(self.test_seq)
		self.cur_doc_i = 0
		self.cur_word_i = 0

	def get_test_batch(self):
		doc_idxs, word_idxs, doc_lengths, context_word_idxs = [],[],[],[]
		learning_rate = self.init_learning_rate * max(0.0001, 
									1.0 - self.finished_word_num / self.words_to_train)
		start_i = self.cur_doc_i
		doc_idx = self.test_seq[self.cur_doc_i]

		while len(doc_idxs) < self.batch_size:
			text_list = self.data_set.doc_text[doc_idx]
			doc_idxs.append(doc_idx)
			doc_lengths.append(1)
			word_idxs.append(text_list[0])
			if self.need_context:
				i = 0
				start_index = i - self.window_size if i - self.window_size > 0 else 0
				context_word_list = text_list[start_index:i] + text_list[i+1:i+self.window_size+1]
				while len(context_word_list) < 2 * self.window_size:
					context_word_list += text_list[start_index:start_index+2*self.window_size-len(context_word_list)]
				context_word_idxs.append(context_word_list)
			
			#move to the next doc
			self.cur_doc_i += 1
			if self.cur_doc_i == len(self.test_seq):
				break
			doc_idx = self.test_seq[self.cur_doc_i]

		batch_context_word_idxs = None
		length = len(word_idxs)
		if self.need_context:
			batch_context_word_idxs = []
			for length_idx in xrange(2 * self.window_size):
				batch_context_word_idxs.append(np.array([context_word_idxs[batch_idx][length_idx]
						for batch_idx in xrange(length)], dtype=np.int64))

		has_next = False if self.cur_doc_i == self.doc_num else True
		return word_idxs, batch_context_word_idxs, doc_idxs, None, doc_lengths, learning_rate, has_next

		
	def single_nce_loss(self, example_vec, label_idxs, label_emb, label_bias,											
						label_size, negative_samples, label_distribution, distort):						
		batch_size = array_ops.shape(label_idxs)[0]#get batch_size		
		#batch_size = tf.Print(batch_size, [batch_size], 'this is batch_size', summarize=5)								
		# Nodes to compute the nce loss w/ candidate sampling.										
		labels_matrix = tf.reshape(tf.cast(label_idxs,dtype=tf.int64),[batch_size, 1])										
												
		# Negative sampling.										
		sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(										
				true_classes=labels_matrix,								
				num_true=1,								
				num_sampled=negative_samples,								
				unique=False,								
				range_max=label_size,								
				distortion=distort,								
				unigrams=label_distribution))																	
												
		#get label embeddings and bias [batch_size, embed_size], [batch_size, 1]										
		true_w = tf.nn.embedding_lookup(label_emb, label_idxs)																			
												
		#get sampled embeddings and bias [num_sampled, embed_size], [num_sampled, 1]										
		sampled_w = tf.nn.embedding_lookup(label_emb, sampled_ids)										
												
		# True logits: [batch_size, 1]										
		true_logits = tf.reduce_sum(tf.multiply(example_vec, true_w), 1)										
												
		# Sampled logits: [batch_size, num_sampled]										
		# We replicate sampled noise lables for all examples in the batch										
		# using the matmul.																				
		sampled_logits = tf.matmul(example_vec, sampled_w, transpose_b=True)

		if self.need_bias:
			true_b = tf.nn.embedding_lookup(label_bias, label_idxs)
			true_logits = true_logits + true_b
			sampled_b = tf.nn.embedding_lookup(label_bias, sampled_ids)
			sampled_b_vec = tf.reshape(sampled_b, [negative_samples])
			sampled_logits = sampled_logits + sampled_b_vec

		return nce_loss(true_logits, sampled_logits), [example_vec, true_w, sampled_w]																		
											
											
def nce_loss(true_logits, sampled_logits):											
	"Build the graph for the NCE loss."										
											
	# cross-entropy(logits, labels)										
	true_xent = tf.nn.sigmoid_cross_entropy_with_logits(										
			logits=true_logits, labels=tf.ones_like(true_logits))								
	sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(										
			logits=sampled_logits, labels=tf.zeros_like(sampled_logits))								
											
	# NCE-loss is the sum of the true and noise (sampled words)																			
	nce_loss_tensor = (tf.reduce_sum(true_xent) + tf.reduce_sum(sampled_xent)) 										
	return nce_loss_tensor			

