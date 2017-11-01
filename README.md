# README #

This is a TensorFlow version of the Enhanced Paragraph Vector. The code is for the training of paragraph vector, so you still need to manually compute the P(w|d) and combine them with the language modeling approach if necessary.
Please cite the following paper if you use this implementation for your research.

*	[1] Qingyao Ai, Liu Yang, Jiafeng Guo, and W. Bruce Croft. 2016. Analysis of the Paragraph Vector Model for Information Retrieval. In Proceedings of the 2016 ACM International Conference on the Theory of Information Retrieval (ICTIR '16). ACM, New York, NY, USA, 133-142. DOI: https://doi.org/10.1145/2970398.2970409
*	[2] Qingyao Ai, Liu Yang, Jiafeng Guo, and W. Bruce Croft. 2016. Improving Language Estimation with the Paragraph Vector Model for Ad-hoc Retrieval. In Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval (SIGIR '16). ACM, New York, NY, USA, 869-872. DOI: https://doi.org/10.1145/2911451.2914688

### Input data: ###
*	vocab.txt.gz: A vocabulary file with one word per line.
*	train.txt.gz: The train data file with a document per line, and the document text is represented using indices of words as they appear in the vocabulary file.

### Training and testing: ###
*	decode=False: train models from the training data.
*	decode=True: 
*		test_mode='output_embedding': output the model embeddings.
*		test_mode='get_doc_softmax_norm': compute the softmax denominators for each document.

In the paper, The final $P_{pv}(w|d)$ is the $exp(w*d)$ divided by the $d$'s denominator.

