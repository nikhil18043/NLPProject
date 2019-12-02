#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 17:37:55 2019

@author: nikhil
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from collections import  defaultdict
import pickle
import pandas as pd
import numpy as np
import operator
import math
import collections
import pandas as pd
from nltk.tokenize import wordpunct_tokenize
import numpy as np
import json
import random
from nltk.tokenize import  word_tokenize 


def plothistogram(data,string):
    plt.figure(figsize=(12,8));
    sns.countplot(data);
    plt.ylabel('Frequency', fontsize=15);
    plt.xlabel(string,fontsize=13);
    plt.xticks(rotation='vertical');
    plt.show();
    
    

training_filename=pd.read_csv('/home/nikhil/Documents/NLP project/js_dataset/programs_training.txt', error_bad_lines=False,header=None)
dictionary=defaultdict(list)
selectfiles=training_filename
path='/home/nikhil/Documents/NLP project/js_dataset/'
selectfiles=np.array(selectfiles)
dataaa=[]

  
# Generates a random number between 
# a given positive range 

def change(daa):
    dataaa=[]
    for i in daa:
            dataaa.append(i)
    return dataaa

for i in range(len(selectfiles)):
    if i%100==0:
        print(i)
    totalsentence=[]
    tempfilename=str(selectfiles[i][0]).strip()
    #file='/home/nikhil/Desktop/Untitled Document.js'
    file=path+tempfilename
    #print(file)
    f = open(file , encoding='latin-1')
    k=f.readlines()
    count=0
    for line in k:
        line=word_tokenize(line)
        count+=len(line)
    dataaa.append(int(count/1000))
dataaa=change(dataaa)

plothistogram(dataaa,'count of words in train data in 1000s')




# a given positive range 


testing_filename=pd.read_csv('/home/nikhil/Documents/NLP project/js_dataset/programs_eval.txt', error_bad_lines=False,header=None)
dictionary=defaultdict(list)
selectfiles=testing_filename
path='/home/nikhil/Documents/NLP project/js_dataset/'
selectfiles=np.array(selectfiles)
dataaa=[]


for i in range(len(selectfiles)):
    if i%100==0:
        print(i)
    totalsentence=[]
    tempfilename=str(selectfiles[i][0]).strip()
    #file='/home/nikhil/Desktop/Untitled Document.js'
    file=path+tempfilename
    #print(file)
    f = open(file , encoding='latin-1')
    k=f.readlines()
    count=0
    for line in k:
        line=word_tokenize(line)
        count+=len(line)
    dataaa.append(int(count/1000))
dataaa=change(dataaa)


plothistogram(dataaa,'count of words in test data in 1000s')

