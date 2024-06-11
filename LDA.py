# Imports

from __future__ import division
import pandas as pd
import gensim
#from gensim.utils import simple_preprocess
#from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
from pprint import pprint
#from gensim import corpora, models
import nltk
import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import os
from vowpalwabbit import pyvw
import string
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.wrappers import LdaVowpalWabbit
import matplotlib.pyplot as plt

eng_stop = [str(word) for word in stopwords.words('english')] 
p_stemmer = PorterStemmer()

#================================ Loading Data =============================================

business_files = []
for i in os.listdir('C:/Users/prasun.j/Dropbox/projects/topic modelling/bbc/business'):
    if i.endswith('.txt'):
        business_files.append(open('C:/Users/prasun.j/Dropbox/projects/topic modelling/bbc/business/'+str(i)))
        
entertainment_files = []
for i in os.listdir('C:/Users/prasun.j/Dropbox/projects/topic modelling/bbc/entertainment'):
    if i.endswith('.txt'):
        entertainment_files.append(open('C:/Users/prasun.j/Dropbox/projects/topic modelling/bbc/entertainment/'+str(i)))
        
politics_files = []
for i in os.listdir('C:/Users/prasun.j/Dropbox/projects/topic modelling/bbc/politics'):
    if i.endswith('.txt'):
        politics_files.append(open('C:/Users/prasun.j/Dropbox/projects/topic modelling/bbc/politics/'+str(i)))
        
sport_files = []
for i in os.listdir('C:/Users/prasun.j/Dropbox/projects/topic modelling/bbc/sport'):
    if i.endswith('.txt'):
        sport_files.append(open('C:/Users/prasun.j/Dropbox/projects/topic modelling/bbc/sport/'+str(i)))
        
tech_files = []
for i in os.listdir('C:/Users/prasun.j/Dropbox/projects/topic modelling/bbc/tech'):
    if i.endswith('.txt'):
        tech_files.append(open('C:/Users/prasun.j/Dropbox/projects/topic modelling/bbc/tech/'+str(i)))
        
total=np.array([business_files,entertainment_files,politics_files,sport_files,tech_files])

data=[] # converting from text to array elements
for i in total:
    for j in i:
        data.append(j.read())
data=np.array(data)

#=============================== Data Preprocessing ================================================

processed_data=[]
tokenizer=RegexpTokenizer(r'\w+')
for i in data:
    lower=i.lower()
    token=tokenizer.tokenize(lower)
    clean=[re.sub(r'[^a-zA-Z]',"",p) for p in token] #replacing any other char with spaces
    while("" in clean):  # for removing space or empty elements 
        clean.remove("")
    stop=[]
    for k in clean: # removing stop words
        if k not in eng_stop:
            stop.append(k)
    processed_data.append(stop)

final_data=[]

for i in processed_data:
    temp=[]
    for j in i:
        if len(j)>3: # removing all words whose length is less than 3
            temp.append(j)
    final_data.append(temp)

flat_list = [item for sublist in final_data for item in sublist]
vocab=list(set(flat_list))

count_dict={} 

for i in vocab: # getting word count
    count_dict[i]=0
    for j in final_data:
        for k in j:
            if i==k:
                count_dict[i]=count_dict[i]+1
                
vocab_rest=[] # creating vocab based on not too infrequent and not too frequent words
for i in count_dict:
    if count_dict[i]<1000:
        if count_dict[i]>100:
            vocab_rest.append(i)

learning_data=[] # making data on new restricted vocab

for i in final_data:
    temp_learning_data=[]
    for j in i:
        if j in vocab_rest:
            temp_learning_data.append(j)
    learning_data.append(temp_learning_data)
    
# Transforming data into IDs
    
text_id=[]
counter=0 # no of documents
for i in learning_data:
    print(counter)
    id_vector=[list(vocab_rest).index(word) for word in i]
    text_id.append(id_vector)
    counter+=1
    
# ================================= LDA ALGORITHM ============================================
   
# LDA Prameters
alpha=0.2 # Alpha is the parameter for the prior topic distribution within documents
beta = 0.01 # Beta is the parameter for the prior topic distribution within documents
K=5 # no of topics
V = len(vocab_rest) # vocab size
D = len(text_id) # no of docs
corpus_itter = 50# iterations

# treat words as columns and values under as sum of occurence in diff categories
word_topic_count = np.zeros((K,V)) # Initialize word-topic count matrix (size KxV,K=topics, V=vocabulary)

# assignment a category to each word of every doc (NA)
topic_doc_assign = [np.zeros(len(sublist)) for sublist in text_id] # Initialize topic-document assignment matrix

# different category constitution of docs
doc_topic_count = np.zeros((D,K)) # Initialize document-topic matrix

for i in range(D):
    for j in range(len(text_id[i])):
        topic_doc_assign[i][j]=np.random.choice(K,1) #np.random.randint(1,5) this also works,choose bet 1/5
        word_id=int(text_id[i][j])
        topic=int(topic_doc_assign[i][j])
    
        word_topic_count[topic,word_id]+=1

for i in range(D):
    for k in range(K):
        topic_doc_vector = topic_doc_assign[i]
        doc_topic_count[i][k]=sum(topic_doc_vector==k)
        
# LDA Algorithm

for itter in range(corpus_itter):
    print(itter)
    for doc in range(D):
        for word in range(len(text_id[doc])):
             
             init_topic_assign = int(topic_doc_assign[doc][word]) # topic of below word
             word_id = text_id[doc][word] # a single word
             
             # Before finding posterior probabilities, remove current word from count matrixes
             doc_topic_count[doc][init_topic_assign] -= 1
             word_topic_count[init_topic_assign][word_id] -=1
             
             # Find probability used for reassigning topics to words within documents
             # Denominator in first term (Numb. of words in doc + numb. topics * alpha)
             denom1 = sum(doc_topic_count[doc]) + K*alpha # float
            
             # Denominator in second term (Numb. of words in topic + numb. words in vocab * beta)
             # sum along rows elements
             denom2 = np.sum(word_topic_count, axis = 1) + V*beta # array
            
             # Numerators, number of words assigned to a topic for every doc
             # Numerator1 = [doc_topic_count[doc][col] for col in range(K)] 
             # above + prior dirichlet param
             #numerator1 = np.array(numerator1) + alpha
             numerator1=doc_topic_count[doc]+alpha # array
             
             # sum of different categories assigned to this word id
             #numerator2 = [word_topic_count[row][word_id] for row in range(K)]
             #numerator2 = np.array(numerator2) + beta
             numerator2=word_topic_count[:,word_id]+beta
             
            
             # Compute conditional probability of assigning each topic
             # Recall that this is obtained from gibbs sampling
             prob_topics = (numerator1/denom1)*(numerator2/denom2)
             prob_topics = prob_topics/sum(prob_topics) # to make sum1 
                                    
             # Update topic assignment (topic can be drawn with prob. found above)
             update_topic_assign = int(np.random.choice(K,1,list(prob_topics)))
             topic_doc_assign[doc][word] = update_topic_assign
             
             # Add in current word back into count matrixes
             doc_topic_count[doc][update_topic_assign] += 1
             word_topic_count[update_topic_assign][word_id] +=1 # [updat..][word_id]?

# document topic distribution            
theta = (doc_topic_count+alpha)
theta_row_sum = np.sum(theta, axis = 1)
theta = theta/theta_row_sum.reshape((D,1))

# Compute posterior mean of word-topic distribution within documents
phi = (word_topic_count + beta)
phi_row_sum = np.sum(phi, axis = 1)
phi = phi/phi_row_sum.reshape((K,1))

#================================= USING LIBRARY =================================            
                
dictionary = gensim.corpora.Dictionary(learning_data)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in learning_data]


Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=200)
print(ldamodel.print_topics(num_topics=5, num_words=10))


#=================================== TESTING =====================================

temp1=[]
temp2=[]               
for i in count_dict.keys():
    if count_dict[i]>6500:
        temp1.append(i)
    if count_dict[i]<100:
        temp2.append(i)