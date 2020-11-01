import os
#conda install -c conda-forge tweepy
import tweepy as tw
import pandas as pd


consumer_key= 'DwAG7jH5CsrYKaFvBuKqWmHK1'
consumer_secret= 'l9DxQb2qysymB0lawL6QlA4ar0lKhhOYVTDRWc3EEvntcoC2Ov'
access_token= '1198471122557263872-WZUHrG9omdMj4atPG3ldVR24HVQROK'
access_token_secret= 'nETxsWZG2PAgi6tunPordZ5E6VyZn91MBYjjGfUprRZUU'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)



#FIND WORDS
search_words = "Utah+Lake -filter:retweets"
date_since = "2015-01-01"
tweets = tw.Cursor(api.search, 
                           q=search_words,
                           lang="en",
                           since=date_since,
                           tweet_mode='extended').items()


users_locs = [[tweet.user.screen_name, tweet.user.location, tweet.full_text, tweet.created_at] for tweet in tweets]
tweet_text = pd.DataFrame(data=users_locs, 
                    columns=['user', "location","content","time"])
#specific phrase
tweet_loc = tweet_text.query('content.str.contains("Utah Lake") or content.str.contains("UtahLake") or content.str.contains("utah lake") or content.str.contains("utahlake")', 
                             engine='python')



#FIND HASTAG
search_words = "#utahlake -filter:retweets"
date_since = "2015-01-01"
tweets = tw.Cursor(api.search, 
                           q=search_words,
                           lang="en",
                           since=date_since,
                           tweet_mode='extended').items()

users_locs = [[tweet.user.screen_name, tweet.user.location, tweet.full_text, tweet.created_at] for tweet in tweets]
hastag = pd.DataFrame(data=users_locs, 
                    columns=['user', "location","content","time"])

  
data = pd.concat([tweet_loc,hastag])

data.to_csv(r'C:\Users\chuon\Desktop\tweet.csv')


