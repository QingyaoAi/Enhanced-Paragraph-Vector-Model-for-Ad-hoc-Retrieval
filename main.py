"""Training and testing the hierarchical embedding model for personalized product search

See the following papers for more information on the hierarchical embedding model.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange	# pylint: disable=redefined-builtin
import tensorflow as tf
import data_util

from ParagraphVector import ParagraphVector


tf.app.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.90,
							"Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 0.0,
							"Clip gradients to this norm.")
tf.app.flags.DEFINE_float("subsampling_rate", 1e-4,
							"The rate to subsampling.")
tf.app.flags.DEFINE_float("distort_rate", 0.75,
							"The distortion rate to vocab sampling.")
tf.app.flags.DEFINE_float("L2_lambda", 0.0,
							"The lambda for L2 regularization.")
tf.app.flags.DEFINE_float("corruption_rate", 0.9,
							"The rate for document corruption.")
tf.app.flags.DEFINE_integer("max_corruption_samples", 100,
							"Maximum number of samples for each corrupted document.")
tf.app.flags.DEFINE_integer("batch_size", 64,
							"Batch size to use during training.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("input_train_dir", "", "The directory of training and testing data")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Model directory & output directory")
tf.app.flags.DEFINE_string("similarity_func", "product", "Select similarity function, which could be product, cosine and bias_product")
tf.app.flags.DEFINE_string("net_struct", "pv_hdc", "Specify network structure parameters. Please read readme.txt for details.")
tf.app.flags.DEFINE_integer("embed_size", 100, "Size of each embedding.")
tf.app.flags.DEFINE_integer("window_size", 3, "Size of context window.")
tf.app.flags.DEFINE_integer("max_train_epoch", 5,
							"Limit on the epochs of training (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
							"How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("seconds_per_checkpoint", 3600,
							"How many seconds to wait before storing embeddings.")
tf.app.flags.DEFINE_integer("negative_sample", 5,
							"How many samples to generate for negative sampling.")
tf.app.flags.DEFINE_boolean("DF_sampling", False, "Set to True for DF based vocab sampling.")
tf.app.flags.DEFINE_boolean("use_local_context", False, "Set to True for Corrupted Doc2Vec to split global context and local context.")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for testing.")
tf.app.flags.DEFINE_string("test_mode", "product_scores", "Test modes: product_scores -> output ranking results and ranking scores; output_embedding -> output embedding representations for users, items and words. (default is product_scores)")
tf.app.flags.DEFINE_integer("rank_cutoff", 100,
							"Rank cutoff for output ranklists.")



FLAGS = tf.app.flags.FLAGS


def create_model(session, forward_only, data_set, doc_num):
	"""Create translation model and initialize or load parameters in session."""
	model = None
	model = ParagraphVector(
		data_set.vocab_size, doc_num, 
		data_set.vocab_distribute,
		FLAGS.window_size, FLAGS.embed_size, FLAGS.max_gradient_norm, FLAGS.batch_size,
		FLAGS.learning_rate, FLAGS.L2_lambda, FLAGS.net_struct,  
		FLAGS.similarity_func, FLAGS.distort_rate, forward_only, FLAGS.negative_sample)
	
	ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
	if ckpt:
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print("Created model with fresh parameters.")
		session.run(tf.global_variables_initializer())
	return model


def train():
	# Prepare data.
	print("Reading data in %s" % FLAGS.data_dir)
	
	data_set = data_util.Tensorflow_data(FLAGS.data_dir, FLAGS.input_train_dir, 'train', FLAGS.DF_sampling)
	data_set.sub_sampling(FLAGS.subsampling_rate)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	#config.log_device_placement=True
	with tf.Session(config=config) as sess:
		# Create model.
		print("Creating model")
		model = create_model(sess, False, data_set, data_set.doc_num)

		print('Start training')
		words_to_train = float(FLAGS.max_train_epoch * data_set.word_count) + 1
		current_words = 0.0
		previous_words = 0.0
		start_time = time.time()
		last_check_point_time = time.time()
		step_time, loss = 0.0, 0.0
		current_epoch = 0
		current_step = 0
		get_batch_time = 0.0
		training_seq = [i for i in xrange(data_set.doc_num)]#[i for i in xrange(10)]
		model.setup_data_set(data_set, words_to_train)
		while True:
			random.shuffle(training_seq)
			model.intialize_epoch(training_seq)
			has_next = True
			while has_next:
				time_flag = time.time()
				word_idxs, context_word_idxs, doc_idxs, doc_word_idxs, doc_lengths, learning_rate, has_next = model.get_train_batch()
				get_batch_time += time.time() - time_flag
				if len(word_idxs) > 0:
					time_flag = time.time()
					step_loss, _ = model.step(sess, learning_rate, word_idxs, context_word_idxs, 
										doc_idxs, doc_word_idxs, doc_lengths, False)
					#step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
					loss += step_loss / FLAGS.steps_per_checkpoint
					current_step += 1
					step_time += time.time() - time_flag

				# Once in a while, we print statistics.
				if current_step % FLAGS.steps_per_checkpoint == 0:
					print("Epoch %d Words %d/%d: lr = %5.3f loss = %6.5f words/sec = %5.2f prepare_time %.2f step_time %.2f\r" %
            				(current_epoch, model.finished_word_num, model.words_to_train, learning_rate, loss, 
            					(model.finished_word_num- previous_words)/(time.time() - start_time), get_batch_time, step_time), end="")
					step_time, loss = 0.0, 0.0
					current_step = 1
					get_batch_time = 0.0
					sys.stdout.flush()
					previous_words = model.finished_word_num
					start_time = time.time()
					#print('time: ' + str(time.time() - last_check_point_time))
					#if time.time() - last_check_point_time > FLAGS.seconds_per_checkpoint:
					#	checkpoint_path_best = os.path.join(FLAGS.train_dir, "ProductSearchEmbedding.ckpt")
					#	model.saver.save(sess, checkpoint_path_best, global_step=model.global_step)

			current_epoch += 1
			#checkpoint_path_best = os.path.join(FLAGS.train_dir, "ProductSearchEmbedding.ckpt")
			#model.saver.save(sess, checkpoint_path_best, global_step=model.global_step)
			if current_epoch >= FLAGS.max_train_epoch:	
				break
		checkpoint_path_best = os.path.join(FLAGS.train_dir, "DocEmbedding.ckpt")
		model.saver.save(sess, checkpoint_path_best, global_step=model.global_step)

def get_doc_softmax_norm():
	# Prepare data.
	print("Reading data in %s" % FLAGS.data_dir)
	
	#if 'pv' in FLAGS.net_struct:
	data_set = data_util.Tensorflow_data(FLAGS.data_dir, FLAGS.input_train_dir, 'train', FLAGS.DF_sampling)
	#else:
	#	data_set = data_util.Tensorflow_data(FLAGS.data_dir, FLAGS.input_train_dir, 'test', FLAGS.DF_sampling)
	current_step = 0
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		# Create model.
		print("Read model")
		model = create_model(sess, True, data_set, data_set.doc_num)
		print('Start softmax denominator computing')
		words_to_train = float(FLAGS.max_train_epoch * data_set.word_count) + 1
		test_seq = [i for i in xrange(data_set.doc_num)]
		model.setup_data_set(data_set, words_to_train)
		model.prepare_test_epoch(test_seq)
		softmax_denominators = []
		has_next = True
		while has_next:
			word_idxs, context_word_idxs, doc_idxs, doc_word_idxs, doc_lengths, learning_rate, has_next = model.get_test_batch()

			if len(word_idxs) > 0:
				doc_softmax_denominator, _ = model.step(sess, learning_rate, word_idxs, context_word_idxs, 
										doc_idxs, doc_word_idxs, doc_lengths, True, FLAGS.test_mode)
				current_step += 1

			# record the results
			for i in xrange(len(doc_idxs)):
				doc_idx = doc_idxs[i]
				softmax_denominators.append((data_set.doc_info[doc_idx][0], doc_softmax_denominator[i]))
				
			if current_step % FLAGS.steps_per_checkpoint == 0:
				print("Finish test doc %d/%d\r" %
            			(model.cur_doc_i, len(model.test_seq)), end="")

	
	with open(FLAGS.train_dir + 'test_doc.softmax_denominators', 'w') as softmax_denominator_fout:
		for i in xrange(len(softmax_denominators)):
			#softmax_denominator_fout.write(softmax_denominators[i][0] + '\t%.3f\n'%softmax_denominators[i][1])
			softmax_denominator_fout.write(softmax_denominators[i][0] + '\t' + str(softmax_denominators[i][1]) + '\n')
	
	return

def output_embedding():
	# Prepare data.
	print("Reading data in %s" % FLAGS.data_dir)
	
	#if 'pv' in FLAGS.net_struct:
	data_set = data_util.Tensorflow_data(FLAGS.data_dir, FLAGS.input_train_dir, 'train', FLAGS.DF_sampling)
	#else:
	#	data_set = data_util.Tensorflow_data(FLAGS.data_dir, FLAGS.input_train_dir, 'test', FLAGS.DF_sampling)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		# Create model.
		print("Read model")
		model = create_model(sess, True, data_set, data_set.doc_num)
		print('Start saving embeddings')
		words_to_train = float(FLAGS.max_train_epoch * data_set.word_count) + 1
		test_seq = [i for i in xrange(data_set.doc_num)]
		model.setup_data_set(data_set, words_to_train)
		model.prepare_test_epoch(test_seq)
		has_next = True
		word_idxs, context_word_idxs, doc_idxs, doc_word_idxs, doc_lengths, learning_rate, has_next = model.get_test_batch()

		part_1 , part_2 = model.step(sess, learning_rate, word_idxs, context_word_idxs, 
									doc_idxs, doc_word_idxs, doc_lengths, True, FLAGS.test_mode)

		# record the results
		word_emb = part_1[0]
		data_set.output_embedding(word_emb, data_set.words, FLAGS.train_dir + 'word_emb.txt')
		if 'pv' in FLAGS.net_struct:
			doc_emb = part_1[1]
			doc_names = [x[0] for x in data_set.doc_info]
			data_set.output_embedding(doc_emb, doc_names, FLAGS.train_dir + 'doc_emb.txt')
			if len(part_2) > 0:
				context_emb = part_2[0]
				data_set.output_embedding(context_emb, data_set.words, FLAGS.train_dir + 'context_emb.txt')
		else: 
			context_emb = part_1[1]
			data_set.output_embedding(context_emb, data_set.words, FLAGS.train_dir + 'context_emb.txt')	
			if FLAGS.use_local_context:
				local_context_emb = part_2[0]
				data_set.output_embedding(local_context_emb, data_set.words, FLAGS.train_dir + 'local_context_emb.txt')	

			#need to compute doc embedding one by one
			words_to_train = float(FLAGS.max_train_epoch * data_set.word_count) + 1
			test_seq = [i for i in xrange(data_set.doc_num)]
			model.setup_data_set(data_set, words_to_train)
			model.prepare_test_epoch(test_seq)
			has_next = True
			current_step = 0
			doc_emb = [None for x in xrange(len(data_set.doc_info))]
			while has_next:
				word_idxs, context_word_idxs, doc_idxs, doc_word_idxs, doc_lengths, learning_rate, has_next = model.get_test_batch()

				if len(doc_idxs) > 0:
					doc_emb_output , _ = model.step(sess, learning_rate, word_idxs, context_word_idxs, 
										doc_idxs, doc_word_idxs, doc_lengths, True, 'output_doc_embedding')
					current_step += 1

				# record the results
				for i in xrange(len(doc_idxs)):
					doc_idx = doc_idxs[i]
					doc_emb[doc_idx] = doc_emb_output[i]
					
				if current_step % FLAGS.steps_per_checkpoint == 0:
					print("Finish test doc %d/%d\r" %
	            			(model.cur_doc_i, len(model.test_seq)), end="")

			doc_names = [x[0] for x in data_set.doc_info]
			data_set.output_embedding(doc_emb, doc_names, FLAGS.train_dir + 'doc_emb.txt')
			
	return

def main(_):
	if FLAGS.input_train_dir == "":
		FLAGS.input_train_dir = FLAGS.data_dir
	if FLAGS.decode:
		if FLAGS.test_mode == 'output_embedding':
			output_embedding()
		elif FLAGS.test_mode == 'get_doc_softmax_norm':
			get_doc_softmax_norm()
	else:
		train()

if __name__ == "__main__":
	tf.app.run()
