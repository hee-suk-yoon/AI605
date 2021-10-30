from datasets import load_dataset
from pprint import pprint
from nltk.tokenize import word_tokenize
import ipdb
import math
from scipy import spatial
import numpy as np
from numpy import dot
from numpy.linalg import norm
squad_dataset = load_dataset('squad')
#Remove duplicates in squad_dataset['validation']
#create vocabulary dictionary
duplicate_rm_dataset = {'id' : [], 'context': [] , 'question': []}
vocab_dict = {}
for idx_, text in enumerate(squad_dataset['validation']['context']):
  #space_tokenized = text.split(' ')
  space_tokenized = word_tokenize(text)
  for idx2_, word in enumerate(space_tokenized):
      word = word.lower()
      if not(word in vocab_dict):
          vocab_dict[word] = 0

doc_vocab_dic = {}
#Go through the document and count the number of times each word appears 
for idx_, data in enumerate(squad_dataset['validation']):
  vocab_dict_temp = vocab_dict.copy()

  space_tokenized = word_tokenize(data['context'])
  for idx2_, word in enumerate(space_tokenized):
    word = word.lower()
    if word in vocab_dict_temp:
      vocab_dict_temp[word] += 1


  #check if the document is a duplicate (There are multiple documents with same 'context' in our validation set). Then only add the document if it is not a duplicate
  
  duplicate = False
  for key, value in doc_vocab_dic.items():
    if value == vocab_dict_temp:
      duplicate = True
      break

  if (not duplicate):
    doc_vocab_dic[data['id']] = vocab_dict_temp
    duplicate_rm_dataset['id'].append(data['id'])
    duplicate_rm_dataset['context'].append(data['context'])
    duplicate_rm_dataset['question'].append(data['question'])
    
    
class SimilaritySearch(object):
  def __init__(self):
    raise NotImplementedError()

  # create the vocab and determine the IDF value of each term. 
  def train(self, documents: list): 
    raise NotImplementedError()

  #Add documents (a list of text)
  # Adding the documents to the index so that they are searchable. 
  def add(self, documents: list):
    raise NotImplementedError()

  #Returns the indices of top-k documents among the added documents
  #that are most similar to the input query 
  def search(self, query: str, k: int) -> list:
    raise NotImplementedError()
  

class BagOfWords(SimilaritySearch):
  def __init__(self):
    self.vocab_dict = {} 
    self.bow = []
    self.IDF = []
    #self.doc_id = []
  def train(self, documents: list):
    word_index = 0
    #create vocabulary dictionary (ok)
    for idx_, document in enumerate(documents):
      space_tokenized = word_tokenize(document)
      for idx2_, word in enumerate(space_tokenized):
          word = word.lower()
          if not(word in self.vocab_dict):
            self.vocab_dict[word] = word_index
            word_index += 1

    
    #iterate through document 
    for doc_id, document in enumerate(documents):
      bow_temp = [0] * len(self.vocab_dict)
      space_tokenized = word_tokenize(document)
      for idx2_, word in enumerate(space_tokenized):
        word = word.lower() #lower case word
        bow_temp[self.vocab_dict[word]] += 1 #increment the word count for this word index 
      self.bow.append(bow_temp)

    #Calculate DF
    DF = []
    for vocab_index in range(0,len(self.vocab_dict)):
      column = [row[vocab_index] for row in self.bow]
      DF.append(len([i for i in column if i > 0]))
    
    #Calculate IDF
    N = len(self.bow) #number of documents
    self.IDF = [math.log(N/x) for x in DF]
     
      
      

class TFIDF (BagOfWords):
  def add(self, documents:list):
    self.documents = []
    for idx_, document in enumerate(documents):
      self.documents.append(document)
    
  def search(self, query: str, k : int) -> list:
    #Calculate TFIDF value for query
    q_space_tokenized = word_tokenize(query)
    q_bow = [0] * len(self.vocab_dict)

    for idx2_, word in enumerate(q_space_tokenized): #get bow for query
      word = word.lower() #lower case word
      if word in self.vocab_dict:
        q_bow[self.vocab_dict[word]] += 1 #increment the word count for this word index 

    q_TF = [x/len(q_space_tokenized) for x in q_bow] #get TF value for query

    q_TFIDF = [t1*t2 for t1,t2 in zip(q_TF,self.IDF)] #calculate TFIDF for query
    
    #self.cossim_value = []
    cossim_value = []
    for idx_, document in enumerate(self.bow):
      #calculate TF value for the document
      num_words = sum(document)
      document_TF = [x/num_words for x in document]
      #calculate TFIDF for the document
      document_TFIDF = [t1*t2 for t1,t2 in zip(document_TF,self.IDF)]
      #perform cosine similarity between q_TFIDF and document_TFIDF
      #cossim_value.append(1-spatial.distance.cosine(q_TFIDF,document_TFIDF))
      a = np.array(document_TFIDF)
      b = np.array(q_TFIDF)
      cossim_value.append(self.cos_sim(a,b))
    
    return sorted(range(len(cossim_value)), key=lambda x: cossim_value[x])[-k:] #return k largest index
  
  def cos_sim(self, A, B):
    return dot(A, B)/(norm(A)*norm(B))
bow = BagOfWords()
bow.train(duplicate_rm_dataset['context'])

tfidf = TFIDF()
tfidf.train(duplicate_rm_dataset['context'])
tfidf.add(duplicate_rm_dataset['context'])
print('start searching')
k_largest_index = tfidf.search(duplicate_rm_dataset['question'][10],10)
correct = 0
for i in range(len(duplicate_rm_dataset['question'])):
  k_largest_index = tfidf.search(duplicate_rm_dataset['question'][i],10)
  if i in k_largest_index:
    correct += 1

correct/len(duplicate_rm_dataset['question'])
print(correct)
#cossim = bow.add(duplicate_rm_dataset['question'][0])
ipdb.set_trace()
