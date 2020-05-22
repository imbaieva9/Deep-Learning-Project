# -*- coding: utf-8 -*-
"""
Created on Thu April  30 20:19:20 2020

@author: imba_ieva
"""

#%%
"""
Importing as many libraries as I can remeber to use
"""
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('stopwords')#In order to use WordCloud
from wordcloud import WordCloud
for dirname, _, filenames in os.walk('C:/Users/imba_ieva/Desktop/iot final project/fake-and-real-news-dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#%%
"""
1.Loading Data for PySpark
As I mentioned below, its not possible to create pipeline using PySpark DF, so we here using Pandas to import csv,
then convert it into PySpark DF 

"""
from pyspark.sql import SparkSession,SQLContext
from pyspark import SparkContext #Reason using sparkcontext instead of session is beacuase Yarn kills my session often when training my model.
sc= SparkContext(master= 'local', appName= 'FakeNewsDetection')
spark= SparkSession(sc)

#Since we are importing 2 datasets its easier to specify a function for them
from pyspark.sql.types import StringType, StructField, StructType
#1. Specify Schema where all columns datatype=string
def read_data(path):
  schema= StructType(
      [StructField('title',StringType(),True),
      StructField('text',StringType(),True),
      StructField('subject',StringType(),True),
      StructField('date',StringType(),True)])
  pd_df= pd.read_csv(path)
  sp_df= spark.createDataFrame(pd_df, schema= schema)
  return sp_df
# Read data set
path_true= "C:/Users/imba_ieva/Desktop/iot final project/fake-and-real-news-dataset/True.csv"
path_fake= "C:/Users/imba_ieva/Desktop/iot final project/fake-and-real-news-dataset/Fake.csv"
true_df= read_data(path_true)
fake_df= read_data(path_fake)

# Old Functions
#fake_df = spark.read.option("quote", "\"") \
#                  .option("escape", "\"") \
#                  .csv("C:/Users/imba_ieva/Desktop/iot final project/fake-and-real-news-dataset/Fake.csv", 
#                       inferSchema=True, sep=',', header=True)
                 
# true_df = spark.read.option("quote", "\"") \
#                  .option("escape", "\"") \
#                  .csv("C:/Users/imba_ieva/Desktop/iot final project/fake-and-real-news-dataset/True.csv", 
#                       inferSchema=True, sep=',', header=True)

true_df.show()
true_df.count()

fake_df.show()
fake_df.count()

#%%

"""
2.Loding Data for Visulization
In order to get some visualizaiton we need pandas DF
"""
fake = pd.read_csv("C:/Users/imba_ieva/Desktop/iot final project/fake-and-real-news-dataset/Fake.csv")
real = pd.read_csv("C:/Users/imba_ieva/Desktop/iot final project/fake-and-real-news-dataset/True.csv")
real.head()
fake.head()

#%%
"""
2.1 Data Visualizaiton
"""

# 1.View fake news data
plt.figure(figsize=(8,5))
sns.countplot("subject", data=fake).set_title('Fake')
plt.show()

# 2.View real news data
plt.figure(figsize=(8,5))
sns.countplot("subject", data=real).set_title('Real')
plt.show()

# 3.Fake news WordCloud
text = ''
for news in fake.text.values:
    text += f" {news}"
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = set(nltk.corpus.stopwords.words("english"))).generate(text)
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
del text

# 4.Real news WordCloud
text = ''
for news in real.text.values:
    text += f" {news}"
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = set(nltk.corpus.stopwords.words("english"))).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
del text

#%% 
"""
3.Data Processing
Concatenate 2 datasets into 1 and shuffle
"""
from pyspark.sql.functions import lit, rand
data= true_df.withColumn('fake', lit(0)).union(fake_df.withColumn('fake', lit(1))).orderBy(rand())
# Check data
data.groupBy('fake').count().show()
# View concatenated result 
data.show(10)

#%%
"""
4.NLP Process
"""
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.feature import StringIndexer, VectorAssembler
StopWordsRemover.loadDefaultStopWords('english')


# 1.Tokenize the title, ignore emoji and etc. regular expression
title_tokenizer= RegexTokenizer(inputCol= 'title', outputCol= 'title_words',pattern= '\\W', toLowercase= True)

# 2.Remove stopwords from title
title_sw_remover= StopWordsRemover(inputCol= 'title_words', outputCol= 'title_sw_removed')

# 3.Compute Term frequency from title
title_count_vectorizer= CountVectorizer(inputCol= 'title_sw_removed', outputCol= 'tf_title')

# 4.Compute TF-IDF from title
title_tfidf= IDF(inputCol= 'tf_title', outputCol= 'tf_idf_title')

# 5.Tokenize the text, ignore emoji and etc. regular expression
text_tokenizer= RegexTokenizer(inputCol= 'text', outputCol= 'text_words',pattern= '\\W', toLowercase= True)

# 6.Remove stopwords from text
text_sw_remover= StopWordsRemover(inputCol= 'text_words', outputCol= 'text_sw_removed')

# 7.Compute Term frequency from text
text_count_vectorizer= CountVectorizer(inputCol= 'text_sw_removed', outputCol= 'tf_text')

# 8.Compute TF-IDF from text
text_tfidf= IDF(inputCol= 'tf_text', outputCol= 'tf_idf_text')

# 9.StringIndexer for subject
subject_str_indexer= StringIndexer(inputCol= 'subject', outputCol= 'subject_idx')

# 10.VectorAssembler
vec_assembler= VectorAssembler(inputCols=['tf_idf_title', 'tf_idf_text', 'subject_idx'], outputCol= 'features')

#%%
"""
5.Building Random Forest Classifier model
"""
from pyspark.ml.classification import RandomForestClassifier
rf= RandomForestClassifier(featuresCol= 'features', labelCol= 'fake', 
                           predictionCol= 'fake_predict', maxDepth= 7, numTrees= 20)

#%%
"""
6.Create Pipeline and fitting data to RF
"""
from pyspark.ml import Pipeline
rf_pipe= Pipeline(stages=[
                title_tokenizer, 
                title_sw_remover, 
                title_count_vectorizer, 
                title_tfidf, 
                text_tokenizer,
                text_sw_remover,
                text_count_vectorizer,
                text_tfidf, 
                subject_str_indexer, 
                vec_assembler,
                rf]) 

clean_data = rf_pipe.fit(data).transform(data)
clean_data.show()
clean_data.take(1)

#Split data into training and test datasets
training, test= data.randomSplit([0.8, 0.2],seed=12345)
# Fitting training data
"""
Ok, So here I've encountered a problem when building pipeline using spark.read.csv, after numerous tries, I've decided to 
go back and import csv using Pandas and Convert it to PySpark DF
"""
rf_model= rf_pipe.fit(training)

#%%
"""
7. Model evaluation
"""
result = rf_model.transform(test)
result.select('fake','fake_predict').show()

#%%
"""
7.1 Confusion matrix
"""
from sklearn.metrics import confusion_matrix
y_true = result.select("fake")
y_true = y_true.toPandas()

y_pred = result.select("fake_predict")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred)
print(cnf_matrix)
print("Prediction Accuracy is ", (cnf_matrix[0,0]+cnf_matrix[1,1])/sum(sum(cnf_matrix)) )
