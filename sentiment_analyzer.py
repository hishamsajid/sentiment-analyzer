import tweepy
import pandas as pn
import simplejson
import textblob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


#key, secret(s), token to be obtained by creating and configuring a twitter app https://apps.twitter.com/
consumer_key = ""
consumer_secret = ""
access_token = ""
access_secret = ""

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)

auth.set_access_token(access_token,access_secret)

api = tweepy.API(auth)

query = "#Jerusalem"

language = "en"

#Extracting useful features from Tweepy Object

users_name  = []
tweet_text = []
location = []
timezone = []
verified = []
favourites_count = []
followers_count = []

for tweet in tweepy.Cursor(api.search, q = query, lang = language).items(2000):
   
    users_name.append(tweet.user.screen_name)
    tweet_text.append(tweet.text)
    location.append(tweet.user.location)
    timezone.append(tweet.user.time_zone)
    verified.append(tweet.user.verified)
    favourites_count.append(tweet.user.favourites_count)
    followers_count.append(tweet.user.followers_count)
    
logs = pn.DataFrame({'user' : pn.Series(users_name), 'tweet' : pn.Series(tweet_text), 'location': pn.Series(location),
'timezone': pn.Series(timezone),'verified' : pn.Series(verified),'favourites_count' : pn.Series(favourites_count),
'followers_count' : pn.Series(followers_count)})

tweets = logs['tweet']


#Using TextBlob to find the polarity of each tweet and then categorizing it
#as either positive, negative or neutral.

polarity = []
sentiment = []
noun_phrases = []
for tweet in tweets.iteritems():
    tweet_str = str(tweet)
    blob = textblob.TextBlob(tweet_str)
    tweet_polarity = blob.sentiment.polarity
    tweet_nounphrases = blob.noun_phrases
    
    if(tweet_polarity == 0):
        sentiment.append('neutral')
    elif(tweet_polarity > 0):
        sentiment.append('positive')
    elif(tweet_polarity < 0):
        sentiment.append('negative')
        
    polarity.append(tweet_polarity)
    noun_phrases.append(tweet_nounphrases)
    
logs['polarity'] = pn.Series(polarity)
logs['sentiment'] = pn.Series(sentiment)
logs['noun phrases'] = pn.Series(noun_phrases)

n_sentiment = logs['sentiment'].value_counts()
print n_sentiment 

#Applying clustering to corroborate our findings via TextBlob

tfidf_vectorizer = TfidfVectorizer(min_df = 1,lowercase = False, ngram_range = (1,1), use_idf = True, stop_words='english')

list_tweets = tweets.tolist()
str_tweets = []
for item in list_tweets:
    str_item = str(item)
    str_tweets.append(str_item)

tfidf_matrix = tfidf_vectorizer.fit_transform(str_tweets)

#Applying K-means clustering with 3 clusters

num_clusters = 3
km = KMeans(n_clusters = num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
series_clusters = pn.Series(clusters)
series_clusters.value_counts()
