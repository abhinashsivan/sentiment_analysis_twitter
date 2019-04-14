import tweepy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from IPython.display import display

consumer_key= 'XXXXXXXXXXXXXXXXXXXXXXXXX'
consumer_secret= 'XXXXXXXXXXXXXXXXXXXXXXXXX'

access_token='XXXXXXXXXXXXXXXXXXXXXXXXX'
access_token_secret='XXXXXXXXXXXXXXXXXXXXXXXXX'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

tweets = api.search('Lucifer', count=200)


data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

display(data.head(10))

print(tweets[0].created_at)
import nltk
nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()


listy = []

for index, row in data.iterrows():
  ss = sid.polarity_scores(row["Tweets"])
  listy.append(ss)
  
se = pd.Series(listy)
data['polarity'] = se.values

display(data.head(100))

