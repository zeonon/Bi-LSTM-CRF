# Bi-LSTM-CRF Tutorial 

### data file 
이 코드를 사용하기 위해서 수집한 HTML file을 json파일로 형식을 바꿔주었다. 

[HTML 파일]()


```
# Written by jiyeon

import os
import json

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

#from nltk.tokenize import WordPunctTokenizer
#from tensorflow.keras.preprocessing.text import text_to_word_sequence



## extract abstracts from json files

path = "/home/jiyeon/tokenization/html_220711145932/data"

dir_list = os.listdir(path)


abstract_list = [ ]


index = 0
for jsonfile in dir_list:
   with open(path+"/"+jsonfile,'r') as f:
       json_data = json.load(f)

   
   try:
      abstract_list.append(json_data["document"]["Abstract"])      

   except:
      print("Error, file name: ", jsonfile)


filename0 = "abstract"

filename_train = "train_test.csv"


f_train = open(filename_train, 'a')

f_train.write("Sentence #,Word,POS,Tag\n")


for i in range(len(abstract_list[100:600])):
#for i in range(500):
   i = i+100
   #print(len(abstract_list[100:600]))
   abstract = abstract_list[i]
   #abstract = abstract.replace('~',' ')
   try:

      words = word_tokenize(abstract)

   except:
      print("Error")
      continue
   tagged = pos_tag(words)
   print(tagged)

   list_of_tuples = tagged

   filename = filename0 + str(i+1) + ".txt"

   f = open(filename, 'w')

   f.write(abstract + "\n")

   f_train.write("Sentence: "+str(i+1))

   for t in list_of_tuples:
      line = ' '.join(str(x) for x in t)
      f.write(line + '\n')
      line_train = ','.join(str(x) for x in t)
      f_train.write(',' + line_train + ',\n')
   f.close()


f_train.close()
   #break
```



### Bi-LSTM
Bi-LSTM+CRF를 하기에 앞서 우선 Bi-LSTM부터 알아보자 



```
import pandas as pd
import numpy as np
import urllib.request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical



data = pd.read_csv("train_600.csv", encoding="latin1", error_bad_lines=False)

print('데이터프레임 행의 개수 : {}'.format(len(data)))
print('데이터에 Null 값이 있는지 유무 : ' + str(data.isnull().values.any()))
print('어떤 열에 Null값이 있는지 출력')
print('==============================')
data.isnull().sum()

print('sentence # 열의 중복을 제거한 값의 개수 : {}'.format(data['Sentence #'].nunique()))
print('Word 열의 중복을 제거한 값의 개수 : {}'.format(data.Word.nunique()))
print('Tag 열의 중복을 제거한 값의 개수 : {}'.format(data.Tag.nunique()))

print('Tag 열의 각각의 값의 개수 카운트')
print('================================')
print(data.groupby('Tag').size().reset_index(name='count'))

data = data.fillna(method="ffill")
print(data.tail())

print(data[:5])




data['Word'] = data['Word'].str.lower()
print('Word 열의 중복을 제거한 값의 개수 : {}'.format(data.Word.nunique()))



func = lambda temp: [(w, t) for w, t in zip(temp["Word"].values.tolist(), temp["Tag"].values.tolist())]
tagged_sentences=[t for t in data.groupby("Sentence #").apply(func)]
print("전체 샘플 개수: {}".format(len(tagged_sentences)))

sentences, ner_tags = [], [] 
for tagged_sentence in tagged_sentences: # 47,959개의 문장 샘플을 1개씩 불러온다.

    # 각 샘플에서 단어들은 sentence에 개체명 태깅 정보들은 tag_info에 저장.
    sentence, tag_info = zip(*tagged_sentence) 
    sentences.append(list(sentence)) # 각 샘플에서 단어 정보만 저장한다.
    ner_tags.append(list(tag_info)) # 각 샘플에서 개체명 태깅 정보만 저장한다.


# 모든 단어를 사용하며 인덱스 1에는 단어 'OOV'를 할당.
src_tokenizer = Tokenizer(oov_token='OOV')
# 태깅 정보들은 내부적으로 대문자를 유지한 채 저장
tar_tokenizer = Tokenizer(lower=False)

src_tokenizer.fit_on_texts(sentences)
tar_tokenizer.fit_on_texts(ner_tags)

word_to_index = src_tokenizer.word_index
index_to_word = src_tokenizer.index_word
ner_to_index = tar_tokenizer.word_index
index_to_ner = tar_tokenizer.index_word
index_to_ner[0] = 'PAD'


print(index_to_ner)


vocab_size = len(src_tokenizer.word_index) + 1
tag_size = len(tar_tokenizer.word_index) + 1
print('단어 집합의 크기 : {}'.format(vocab_size))
print('개체명 태깅 정보 집합의 크기 : {}'.format(tag_size))

X_data = src_tokenizer.texts_to_sequences(sentences)
y_data = tar_tokenizer.texts_to_sequences(ner_tags)

max_len = 70
X_data = pad_sequences(X_data, padding='post', maxlen=max_len)
y_data = pad_sequences(y_data, padding='post', maxlen=max_len)

X_train, X_test, y_train_int, y_test_int = train_test_split(X_data, y_data, test_size=.2, random_state=777)

y_train = to_categorical(y_train_int, num_classes=tag_size)
y_test = to_categorical(y_test_int, num_classes=tag_size)
```


### CRF
```
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Bidirectional, TimeDistributed, Embedding, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_crf import CRFModel
from seqeval.metrics import f1_score, classification_report


embedding_dim = 128
hidden_units = 64    
dropout_ratio = 0.3  



sequence_input = Input(shape=(max_len,),dtype=tf.int32, name='sequence_input')

model_embedding = Embedding(input_dim=vocab_size,
                            output_dim=embedding_dim,
                            input_length=max_len)(sequence_input)

model_bilstm = Bidirectional(LSTM(units=hidden_units, return_sequences=True))(model_embedding)

model_dropout = TimeDistributed(Dropout(dropout_ratio))(model_bilstm)

model_dense = TimeDistributed(Dense(tag_size, activation='relu'))(model_dropout)

base = Model(inputs=sequence_input, outputs=model_dense)
model = CRFModel(base, tag_size)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), metrics='accuracy')




es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('bilstm_crf/cp.ckpt', monitor='val_decode_sequence_accuracy', mode='max', verbose=1, save_best_only=True, save_weights_only=True)

history = model.fit(X_train, y_train_int, batch_size=128, epochs=15, validation_split=0.1, callbacks=[mc, es])




model.load_weights('bilstm_crf/cp.ckpt')

i = 0  # 확인하고 싶은 테스트용 샘플의 인덱스.
y_predicted = model.predict(np.array([X_test[i]]))[0] # 입력한 테스트용 샘플에 대해서 예측 y를 리턴
labels = np.argmax(y_test[i], -1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경.

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35 * "-")


for word, tag, pred in zip(X_test[i], labels, y_predicted[0]):
    if word != 0: # PAD값은 제외함.
        print("{:17}: {:7} {}".format(index_to_word[word], index_to_ner[tag], index_to_ner[pred]))



f_out = open("output.txt", 'w')
for i in range(len(X_train[:,0])):
    y_predicted = model.predict(np.array([X_train[i]]))[0]  # 입력한 테스트용 샘플에 대해서 예측 y를 리턴
    y_predicted = np.argmax(y_predicted, axis=-1) # 확률 벡터를 정수 인코딩으로 변경함.
    labels = np.argmax(y_train[i], -1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.

    f_out.write("[Train] Abstract: "+str(i+1)+"\n")

    f_out.write("{:15}|{:5}|{}\n".format("단어", "실제값", "예측값"))
    f_out.write(35 * "-")
    f_out.write("\n")

    for word, tag, pred in zip(X_train[i], labels, y_predicted[0]):
        if word != 0: # PAD값은 제외함.
           f_out.write("{:17}: {:7} {}\n".format(index_to_word[word], index_to_ner[tag], index_to_ner[pred]))


for i in range(len(X_test[:,0])):
    y_predicted = model.predict(np.array([X_test[i]]))[0] # 입력한 테스트용 샘플에 대해서 예측 y를 리턴
    y_predicted = np.argmax(y_predicted, axis=-1) # 확률 벡터를 정수 인코딩으로 변경함.
    labels = np.argmax(y_test[i], -1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.

    f_out.write("[Test] Abstract: "+str(i)+"\n")

    f_out.write("{:15}|{:5}|{}\n".format("단어", "실제값", "예측값"))
    f_out.write(35 * "-")
    f_out.write("\n")

    for word, tag, pred in zip(X_test[i], labels, y_predicted[0]):
        if word != 0: # PAD값은 제외함.
           f_out.write("{:17}: {:7} {}\n".format(index_to_word[word], index_to_ner[tag], index_to_ner[pred]))

f_out.close()
```   




### F1-score
```
def sequences_to_tag_for_crf(sequences): 
    result = []
    # 전체 시퀀스로부터 시퀀스를 하나씩 꺼낸다.
    for sequence in sequences: 
        word_sequence = []
        # 시퀀스로부터 예측 정수 레이블을 하나씩 꺼낸다.
        for pred_index in sequence:
            # index_to_ner을 사용하여 정수를 태깅 정보로 변환. 'PAD'는 'O'로 변경.
            pred_index = np.argmax(pred)
            word_sequence.append(index_to_ner[pred_index].replace("PAD", "O"))
        result.append(word_sequence)
    return result


y_predicted = model.predict(X_test)[0]
pred_tags = sequences_to_tag_for_crf(y_predicted)
test_tags = sequences_to_tag(y_test)

print("F1-score: {:.1%}".format(f1_score(test_tags, pred_tags)))
print(classification_report(test_tags, pred_tags))


y_predicted = model.predict(X_train)[0]
test_tags = sequences_to_tag_for_crf(y_test)
pred_tags = sequences_to_tag_for_crf(y_predicted)
train_tags = sequences_to_tag(y_train)

print("F1-score: {:.1%}".format(f1_score(train_tags, pred_tags)))
print(classification_report(train_tags, pred_tags))
```







