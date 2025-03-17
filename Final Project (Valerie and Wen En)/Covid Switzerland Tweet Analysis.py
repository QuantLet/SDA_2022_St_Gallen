# ## Import Packages
import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
import string
import textblob
from textblob import TextBlob
import wordcloud
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

# ## Gathering Data
# Creating list to append tweet data to
tweets_list = []

# search terms
search_terms = 'covid'

# Using TwitterSearchScraper to scrape data and append tweets to list
for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_terms + ' since:2020-04-07 until:2022-12-10 near:"Switzerland" within:10km').get_items()):
    if i > 10000:
        break
    #declare the attributes to be returned
    tweets_list.append([tweet.content])

# Creating a dataframe from the tweets list above
tweets_df = pd.DataFrame(tweets_list, columns=['Text'])
tweets_df.to_csv("covid.csv")

# Read data
covid_data = pd.read_csv("covid.csv")

covid_data = covid_data.iloc[:,1:]

# ## Data Preprocessing
covid_data.head()

# Check for missing data
covid_data.isna().sum()
covid_data.describe()
covid_data.info()

# ## Sentiment Analysis
# Clean Tweet by removing links, special characters
def clean_tweet(tweet): 
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+://\S+)", " ", tweet).split())

# Classify sentiment using the textblob method
def analyse_tweet(tweet):
    cleaned_tweets = clean_tweet(tweet) 
    tweet_sentiment = TextBlob(tweet).sentiment.polarity
    return tweet_sentiment

def get_tweet_sentiment(tweet): 
    tweet_polarity = analyse_tweet(tweet) 
    if tweet_polarity >0:
        return 'positive'
    elif tweet_polarity == 0:
        return 'neutral'
    else:
        return 'negative'

covid_data['Sentiment'] = covid_data['Text'].apply(lambda x: get_tweet_sentiment(x))
covid_data['Sentiment'].value_counts().plot(kind='bar')

# ## Word cloud
def wordcloud(string):
    wc = WordCloud(background_color=color, width=1200,height=600,mask=None,random_state=1,
                   max_font_size=200,stopwords=stop_words,collocations=False).generate(string)
    fig=plt.figure(figsize=(20,8))
    plt.axis('off')
    plt.title('--- WordCloud for {} --- '.format(title),weight='bold', size=30)
    plt.imshow(wc)


covid_data['Text'] = covid_data['Text'].apply(lambda x: clean_tweet(x))

stop_words=set(STOPWORDS)
covid_data_string = " ".join(covid_data['Text'].astype('str'))

covid_data.head()

# create the wordcloud
tweet_string  = " ".join(tweet for tweet in covid_data["Text"])
from wordcloud import WordCloud   # for the wordcloud :)
tweet_wordcloud = WordCloud(background_color="white", max_words=100).generate(tweet_string)

# view the wordcloud
import matplotlib.pyplot as plt   # for wordclouds & charts
plt.imshow(tweet_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

