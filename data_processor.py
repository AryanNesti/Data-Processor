#!/usr/bin/env python
# coding: utf-8

# In[9]:




# mount your Google Drive, so that you can read data from it.
# Note: it needs your authorization.
# from google.colab import drive
# drive.mount('/content/drive')


# In[27]:


import numpy as np
import random
from tqdm import tqdm
import pandas as pd 


# **Utility Functions**

# In[3]:


import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import *

# stemming tool from nltk
stemmer = PorterStemmer()
# a mapping dictionary that help remove punctuations
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def get_tokens(text):
    # turn document into lowercase
    lowers = text.lower()
    # remove punctuation
    no_punctuation = lowers.translate(remove_punctuation_map)
    # tokenize document
    tokens = nltk.word_tokenize(no_punctuation)
    # stop words
    filtered = [w for w in tokens if not w in stopwords.words("english")]
    # stemming process
    stemmed = []
    for item in filtered:
        stemmed.append(stemmer.stem(item))

    return stemmed


# In[4]:


import numpy as np


def get_dict(fpath):
    dictionary = {}


    with open(fpath, "r") as f:
        for i, word in enumerate(f):
            dictionary[word.strip()] = i

    return dictionary


def get_doc_tf(word_set, dictionary):
    n_words = len(dictionary)
    tf_vec = np.zeros(n_words)

    max_cnt = 0
    for word in word_set:
        idx = dictionary[word]
        tf_vec[idx] += 1.0

        if tf_vec[idx] > max_cnt:
            max_cnt = tf_vec[idx]

    return tf_vec / max_cnt



def get_tf_idf(tf_dict, df_vec, n_doc, n_words):

    tf_idf_mtx = np.zeros((n_doc, n_words))
    idf = np.log(n_doc / df_vec)

    for doc_idx, tf_vec in tf_dict.items():
        tf_idf = tf_dict[doc_idx]*idf

        tf_idf_mtx[doc_idx, :] = tf_idf

    return tf_idf_mtx


def write(d, fpath):

    with open(fpath, "w") as f:

        for k, v in d.items():

            f.write(f"{k}\n")
            


def filter_top_k(counter_sorted, limit):
    top_k = {}

    for i, k in enumerate(counter_sorted.keys()):
        if i == limit:
            break
        top_k[k] = counter_sorted[k]

    return top_k


# In[ ]:





# In[ ]:





# **Compute TF-IDF Matrix**

# In[5]:


def tfidf_main(fpath, dictionary):


    n_words = len(dictionary)
    tf = {}
    doc_freq = np.zeros(n_words)

    with open(fpath, 'r') as f:

        lines = f.readlines()
        n_doc = len(lines) - 1

        for i, line in tqdm(enumerate(lines), total=n_doc+1):
            if i == 0:
                continue

            doc_idx = i - 1

            id, txt, cat = line.split(",")
            cat = cat.strip()
            tokens = get_tokens(txt)

            filtered = []
            filtered_unique = set()
            for word in tokens:
                if word in dictionary:
                    filtered.append(word)
                    filtered_unique.add(word)

            # get term frequency
            tf_vec = get_doc_tf(filtered, dictionary)
            tf[doc_idx] = tf_vec

            # get doc frequency:
            for word in filtered_unique:
                idx = dictionary[word]
                doc_freq[idx] += 1


    tfidf_mtx = get_tf_idf(tf, doc_freq, n_doc, n_words)


    return tfidf_mtx


# In[19]:


dictionary = get_dict("dictionary.txt")
tfidf = tfidf_main("news-train.csv", dictionary)
np.savetxt("tfidf.txt", tfidf,  fmt='%.4f', delimiter=",")


# In[ ]:


tfidf


# **Word Frequencies**

# In[13]:


def frequency_main(limit, fpath, dictionary):



    with open(fpath, 'r') as f:

        lines = f.readlines()
        n_doc = len(lines) - 1

        stratifed_cntr = {
                        "sport": {},
                        "business": {},
                        "politics": {},
                        "entertainment": {},
                        "tech": {}
                    }


        for i, line in tqdm(enumerate(lines), total=n_doc + 1):
            if i == 0:
                continue

            id, txt, cat = line.split(",")
            cat = cat.strip()
            tokens = get_tokens(txt)

            for t in tokens:
                if t not in dictionary:
                    continue

                if t not in stratifed_cntr[cat]:
                    stratifed_cntr[cat][t] = 0

                stratifed_cntr[cat][t] += 1

        stratifed_sorted = {}
        for cat, cnts in stratifed_cntr.items():
            stratifed_sorted[cat] = {k: v for k, v in sorted(cnts.items(), key=lambda item: item[1], reverse=True)}


        stratified_output = {}
        for cat, cnts in stratifed_sorted.items():
            stratified_output[cat] = filter_top_k(cnts, limit)

    return stratified_output


# In[20]:


counts = frequency_main(limit=3, fpath="news-train.csv", dictionary=dictionary)


# In[21]:


counts


# In[ ]:





# **Average TFIDF Scores by Category**

# In[16]:


def mean_tfidf_main(trn_fpath, tfidf_fpath, dictionary, k):

    idx_to_word = {}
    for key, val in dictionary.items():
        idx_to_word[val] = key

    with open(trn_fpath, 'r') as f:

        lines = f.readlines()
        n_doc = len(lines) - 1
        cats = np.zeros((n_doc, 1), dtype=object)

        for i, line in tqdm(enumerate(lines), total=n_doc + 1):
            if i == 0:
                continue

            doc_idx = i - 1
            id, txt, cat = line.split(",")
            cat = cat.strip()

            cats[doc_idx, 0] = cat

        tfidf = np.loadtxt(tfidf_fpath, delimiter=",")

        df = pd.DataFrame(np.concatenate([cats, tfidf], axis=1))

        groups = df.groupby(0)

        output = {}

        for cat, chunk in groups:

            mean = chunk.values[:, 1:].mean(axis=0)

            word_idx = np.argsort(mean)
            s = np.sort(mean)
            top_k = word_idx[-k:]

            output[cat] = {}
            for idx in top_k:
                word = idx_to_word[idx]
                score = mean[idx]
                #record = {"word": word, "score": score}
                output[cat][word] = score


    return output


# In[29]:


avg_tfidf = mean_tfidf_main("news-train.csv", "tfidf.txt", dictionary, k=3)


# In[30]:


avg_tfidf


# In[ ]:




