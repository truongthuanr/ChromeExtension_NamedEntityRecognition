import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model,load_model
# from datasets import load_dataset

import os
from keras.layers import Input,Embedding,TimeDistributed,\
                         Dropout,Conv1D,MaxPooling1D,\
                         Flatten,Bidirectional,LSTM,Dense,\
                         concatenate

from keras.initializers import RandomUniform

from keras.optimizers import Adam



# ---------------------------------------------------------------
#      Configuration parameters
# ---------------------------------------------------------------
max_word_tokens = 24000
max_sentence_length = 50
max_word_len = 20

label2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
id2label = {v: k for k, v in label2id.items()}

# ---------------------------------------------------------------
#     CHARS PREPROCESSING
# ---------------------------------------------------------------
char2Idx = {"PADDING":0, "UNKNOWN":1}
for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"+\
         ".,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
    char2Idx[c] = len(char2Idx)
len(char2Idx)

def character_vectorize(X):
  data_vec = []
  for sentence in X:
    # print(f"Sentence: {sentence}")
    padchar = char2Idx['PADDING']
    sentence_vec=[]
    for word in sentence:
      # print(word)
      # if unknownchar in word
      chars = []
      if len(word) >= max_word_len:

        chars=[char2Idx.get(c,1) for c in word[:max_word_len]]
      else:
        prepad=int((max_word_len-len(word))/2)
        postpad=max_word_len-(len(word)+prepad)
        chars.extend([padchar]*prepad)
        chars.extend([char2Idx.get(c,1) for c in word])
        chars.extend([padchar]*postpad)
      sentence_vec.append(chars)

    data_vec.append(sentence_vec)
  data_vec = np.asarray(data_vec, dtype=object)
  return data_vec

def char_preprocessing(char_input):
  '''
  input:: char_input: list of words
  output: vecterized array of character
  with shape(sentences, max_sentence_len, max_word_len)
  '''
  X_chars = character_vectorize(char_input)
  X_chars = pad_sequences(sequences = X_chars,
                          maxlen=max_sentence_length,
                          dtype=object,
                          padding="post",
                          truncating="post",
                          value=0)

  X_chars = np.asarray(X_chars,
                       dtype=np.float32)

  return X_chars

# ---------------------------------------------------------------
#     CASE PREPROCESSING
# ---------------------------------------------------------------
case2id = {'allcaps':0,
           'upperinitial':1,
           'lower':2,
           'mixedcaps':3,
           'noinfo':4}
id2case = {v:k for k,v in case2id.items()}

def case_vectorize(input):
  '''
  input: array of sentences, sentencs is list of word
  '''
  case_vec = []

  for sentence in input:
    sen_case_type = []
    for word in sentence:
      temp = [0]*len(case2id)
      # if word is Title
      if word.istitle():
        temp[case2id['upperinitial']] =1
        sen_case_type.append(temp)

      # if uper, lower, mixed or else
      else:
        if word.isupper():
          temp[case2id['allcaps']] = 1
          sen_case_type.append(temp)
        elif word.islower():
          temp[case2id['lower']]=1
          sen_case_type.append(temp)
        else:
          if word.lower().islower():
            temp[case2id['mixedcaps']]=1
            sen_case_type.append(temp)
          else:
            temp[case2id['noinfo']]=1
            sen_case_type.append(temp)


    case_vec.append(sen_case_type)
  return case_vec

def case_preprocesing(X_input):
  paddingnoinfo = [0,0,0,0,1]
  X_case = case_vectorize(X_input)

  X_case = pad_sequences(sequences = X_case,
                        maxlen=max_sentence_length,
                        dtype=object,
                        padding="post",
                        truncating="post",
                        value=paddingnoinfo)

  X_case = np.asarray(X_case,
                      dtype=np.float32)

  return X_case



# ---------------------------------------------------------------
#     MAKE PREDICTION
# ---------------------------------------------------------------
print("============================================")
path = "\\ner\\model\\bilstm-cnn"
cwd = os.getcwd()
path = os.path.join(cwd,'ner\\model\\bilstm-cnn')
print(path)
loaded_model = tf.keras.saving.load_model(path)

def make_prediction(model, input_string):
  X_words = np.asarray([input_string])
  X_input_list = [input_string.split()]
  X_chars = char_preprocessing(X_input_list)
  X_case = case_preprocesing(X_input_list)
  predictions = model.predict([X_words,X_chars,X_case])

  return predictions


def string2tag(input_string):
  n = len(input_string.split())
  predictions = make_prediction(loaded_model,input_string)
  predictions = np.squeeze(np.argmax(predictions,axis=2))
  predictions = [id2label[word] for word in predictions]
  
  result = [f"{input_string.split()[i]}<{predictions[i]}>" for i in range(n)]
  result = "\n".join(result)

  return result