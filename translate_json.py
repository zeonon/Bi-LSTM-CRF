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



