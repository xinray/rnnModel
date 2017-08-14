import sys
import numpy as np
import re
import itertools
import codecs
from collections import Counter
import pandas as pd
from keras.utils.np_utils import to_categorical

### train data
def load_data_and_labels(path):
    data_train = pd.read_csv(path, sep='\t')

    x_text = []
    y = []
    for idx in range(data_train.review.shape[0]):
        text = (data_train.review[idx])

        sentences = text.split()

        x_text.append(sentences)
        
        y.append(data_train.sentiment[idx])

    #y = to_categorical(y)  

    return [x_text, y]

### test data
def load_x_data(path):
    data_train = pd.read_csv(path, sep='\t')

    x_text = []
    y = []
    for idx in range(data_train.review.shape[0]):
        text = (data_train.review[idx])

        sentences = text.split()

        x_text.append(sentences)
        
    return x_text

def pad_sentences(sentences, maxlen, padding_word="<PAD/>"):
  sequence_length = max(len(x) for x in sentences)
  sequence_length = maxlen

  padded_sentences = []
  for i in range(len(sentences)):
    sentence = sentences[i]
    #num_padding = sequence_length - len(sentence)
    num_padding = 0
    new_sentence = sentence + [padding_word] * num_padding
    padded_sentences.append(new_sentence)

  return padded_sentences

def build_vocab(sentences):
  """
  Builds a vocabulary mapping from word to index based on the sentences.
  Returns vocabulary mapping and inverse vocabulary mapping.
  """

  vocabulary_inv = []
  vocabulary_inv.append('<PAD/>')
  vocabulary_inv.append('UNK')
  for line in open('data/stopwords.txt') :
    line = line.strip()
    sep = line.split()

    word = sep[0]
    cnt = int(sep[1])

    if cnt < 5:
        continue

    vocabulary_inv.append(word)

  # Mapping from word to index
  vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

  return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
  """
  Maps sentencs and labels to vectors based on a vocabulary.
  x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
  y = np.array(labels)
  """

  x = build_x_input_data(sentences, vocabulary) 
  y = build_y_input_data(labels)

  return [x, y]

def build_x_input_data(sentences, vocabulary):
  """
  Maps sentencs and labels to vectors based on a vocabulary.
  x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
  """
  x = np.array([[vocabulary['UNK'] if(word not in vocabulary) else vocabulary[word] for word in sentence] for sentence in sentences])

  return x

def build_y_input_data(labels):
  """
  Maps sentencs and labels to vectors based on a vocabulary.
  """
  y = np.array(labels)

  return y

###load train data
def inner_load_data(path,sequence_length, vocabulary):
  """
  Loads and preprocessed data for the MR dataset.
  Returns input vectors, labels, vocabulary, and inverse vocabulary.
  """

  # Load and preprocess data
  sentences, labels = load_data_and_labels(path)
  sentences_padded = pad_sentences(sentences, sequence_length)
  print "sequence_length: ", sequence_length
  
  vocabulary_inv = None
  if vocabulary == None:
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)

  x, y = build_input_data(sentences_padded, labels, vocabulary)
  return [x, y, vocabulary, vocabulary_inv]

def train_test_split(X, y, train_size):
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = X[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    tlen = int(len(y)*train_size)
    train_X = x_shuffled[:tlen]
    train_y= y_shuffled[:tlen]
    test_X = x_shuffled[tlen:]
    test_y = y_shuffled[tlen:]

    #train_X = x_shuffled[:len(y)*train_size]
    #train_y= y_shuffled[:len(y)*train_size]
    #test_X = x_shuffled[len(y)*train_size:]
    #test_y = y_shuffled[len(y)*train_size:]
    return train_X, test_X, train_y, test_y

def load_data(path, max_len, batch_size, valid_portion=0.1, voc=None):
    train_set_x, train_set_y, vocabulary, vocabulary_inv = inner_load_data(path, max_len, voc)

    #shuffle and generate train and valid dataset
    if valid_portion < 1:
        train_set_x, valid_set_x, train_set_y, valid_set_y = train_test_split(train_set_x, train_set_y, valid_portion)
    else:
        valid_set_x = []
        valid_set_y = []
        
    if len(train_set_x) != len(train_set_y) or len(valid_set_x) != len(valid_set_y):
        print "train x sex len != train y set len"
        sys.exit(0) 

    new_train_set_x = np.zeros([len(train_set_x),max_len])
    new_train_set_x = np.array([[0] * max_len] * len(train_set_x))
    new_train_set_x_mask = np.zeros([len(train_set_x),max_len])
    new_train_set_y = np.zeros(len(train_set_y))

    new_valid_set_x = np.zeros([len(valid_set_x),max_len])
    new_valid_set_x_mask = np.zeros([len(valid_set_x),max_len])
    new_valid_set_y = np.zeros(len(valid_set_y))

    def padding_and_generate_mask(inx, iny, new_x, new_y, new_mask_x):

        for i,(x,y) in enumerate(zip(inx,iny)):
            if len(x) <= max_len:
                new_x[i, 0:len(x)] = x
                new_mask_x[i, 0:len(x)] = 1
                new_y[i] = y
            else:
                new_x[i] = (x[0:max_len])
                new_mask_x[i, 0:max_len] = 1
                new_y[i] = y

        print "new_x:", len(new_x)
        print "new_y:", len(new_y)
        new_set =(new_x, new_y, new_mask_x)

        del new_x, new_y, new_mask_x
        return new_set

    train_set = padding_and_generate_mask(train_set_x, train_set_y, new_train_set_x, new_train_set_y, new_train_set_x_mask)
    valid_set = padding_and_generate_mask(valid_set_x, valid_set_y, new_valid_set_x, new_valid_set_y, new_valid_set_x_mask)

    return train_set, valid_set, vocabulary, vocabulary_inv

#return batch dataset
def batch_iter(data, batch_size):
    x, y, mask_x = data
    x = np.array(x)
    y = np.array(y)
    data_size = len(x)

    num_batches_per_epoch = int((data_size-1)/batch_size)
    #print ("num_batches_per_epoch:%s, batch_size:%s, data_size:%s" % (num_batches_per_epoch, batch_size, data_size))

    for batch_index in range(num_batches_per_epoch):
        start_index = batch_index*batch_size
        end_index = min((batch_index+1)*batch_size, data_size)

        return_x = x[start_index:end_index, :]
        return_y = y[start_index:end_index]
        return_mask_x = mask_x[start_index:end_index, : ]

        yield (return_x, return_y, return_mask_x, num_batches_per_epoch)
