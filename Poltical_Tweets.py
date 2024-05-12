# -*- coding: utf-8 -*-

#importing library
import pandas as pd

# retrieving data from csv file and storing it into a dataframe
twitter_data=pd.read_csv('path')
twitter_data.head()


# displaying shape
twitter_data.shape


# checking for missing values
twitter_data.isna().sum()


# dropping rows with missing values
twitter_data = twitter_data.dropna()

# displaying new shape
twitter_data.shape


# importing libraries
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# creating a vectorizer object using 1-grams and 2-grams
vectorizer = CountVectorizer(ngram_range=(1, 2))

# encoding the corpus
# extracting token counts out of raw text documents using the vocabulary
token_count_matrix = vectorizer.fit_transform(twitter_data['clean_text'])

# summarizing the numerical features from texts
print(f'The size of the feature matrix for the texts = {token_count_matrix.get_shape()}')
print(f'The first row of the feature matrix = {token_count_matrix[0, ]}.')
print(f'There are {token_count_matrix[0, ].count_nonzero()}/{token_count_matrix.get_shape()[1]} non-zeros')


# importing library
from sklearn.feature_extraction.text import TfidfTransformer

# creating a vectorizer object using the default parameters
vectorizer = CountVectorizer()

# extracting token counts out of raw text documents using the vocabulary
token_count_matrix = vectorizer.fit_transform(twitter_data['clean_text'])

# summarizing token count matrix
print(f'The size of the count matrix for the texts = {token_count_matrix.get_shape()}')
print(f'The sparse count matrix is as follows:')
print(token_count_matrix)

# creating a tf_idf object using the default parameters
tf_idf_transformer=TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=False)

# fitting to the token_count_matrix, then transforming it to a normalized tf-idf representation
tf_idf_matrix_1 = tf_idf_transformer.fit_transform(token_count_matrix)

# summarizing the tf_idf_matrix
print(f'The size of the tf_idf matrix for the texts = {tf_idf_matrix_1.get_shape()}')
print(f'The sparse tf_idf matrix is as follows:')
print(tf_idf_matrix_1)

"""## 4. Perform the tf-idf analysis on the column of the clean_text using Tfidfvectorizer.

We will do the same thing using only the Tfidfvectorizer.
"""

#importing library
from sklearn.feature_extraction.text import TfidfVectorizer

# creating a TfidfVectorizer Object using the default parameters
tfidf_vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=False)

# fitting to the corpus, then converingt a collection of raw documents to a matrix of TF-IDF features.
tf_idf_matrix_2 = tfidf_vectorizer.fit_transform(twitter_data['clean_text'])

# summarizing the tf_idf_matrix
print(f'The size of the tf_idf matrix for the texts = {tf_idf_matrix_2.get_shape()}')
print(f'The sparse tf_idf matrix is as follows:')
print(tf_idf_matrix_2)


#importing library
from sklearn.feature_extraction.text import HashingVectorizer

#creating a HashingVectorizer object using the default parameters
hash_vectorizer = HashingVectorizer()

# converting a collecting of text documents to a matrix token counts using hash vectorizing
token_count_matrix=hash_vectorizer.fit_transform(twitter_data['clean_text'])

# summarizing the count matrix
print(f'The size of the count matrix for the texts = {token_count_matrix.get_shape()}')
print(f'The sparse count matrix is as follows:')
print(token_count_matrix)

# we will use the transformer we created in step 3 since it is already set to the default parameters
# fitting to the count matrix, then transforming it to a normalized tf-idf representation
tf_idf_matrix_3 = tf_idf_transformer.fit_transform(token_count_matrix)

# summarizing the tf_idf_matrix
print(f'The size of the tf_idf matrix for the texts = {tf_idf_matrix_3.get_shape()}')
print(f'The sparse tf_idf matrix is as follows:')
print(tf_idf_matrix_3)
