#load packages
from typing import Text
import pandas as pd
import numpy
from textblob import TextBlob

#load data viz pkgs
import matplotlib.pyplot as plt
import seaborn as sns

#text cleaning
import neattext.functions as nfx

#load dataset
df = pd.read_csv('C:/Users/Administrator/Desktop/sentimentanalysis/emotion_dataset_2.csv')

#preview
#df.head()

#shape
df.shape

#datatypes
df.dtypes

#check for missing values
df.isnull().sum()

#value count of emotions
#df['Emotion'].value_counts

#value count of emotions
#df['Emotion'].value_counts().plot(kind='bar')
#plt.show()

#using seaborn to plot
#sns.countplot(df['Emotion'])

#new method

#sns.countplot(x='Emotion', data=df)
#plt.show()


#SentimentAnalysis
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    if sentiment > 0:
        result = "Positive"
    elif sentiment < 0:
        result = "Negative"
    else: 
        result = "Neutral"    
    return result   
    
#text fxn
#get_sentiment("I love coding")

df['Sentiment'] = df['Text'].apply(get_sentiment)
df.head()

#Method using MatPlotLib
#compare emotions vs sentiment
#df.groupby(['Emotion','Sentiment']).size().plot(kind='bar')

#using seaborn
plt.figure(figsize=(10,5))
sns.catplot(x='Emotion',hue='Sentiment',data=df,kind='count',height=6,aspect=1.5)
plt.show()



