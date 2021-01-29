# Databricks notebook source
# MAGIC %pip install -U pandasql

# COMMAND ----------

# MAGIC %pip install WordCloud

# COMMAND ----------

# MAGIC %pip install TextBlob

# COMMAND ----------

import pandas as pd
import numpy as np
import ast
import pprint
import json
import math
import glob
from time import sleep
from functools import reduce
from matplotlib import pyplot as plt

from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

# COMMAND ----------

# File location and type
file_location = "/user/hive/warehouse/tweets"
file_type = "parquet"

# CSV options
#infer_schema = "true"

# The applied options are for CSV files. For other file types, these will be ignored.
s_cg_tweets = spark.read.format(file_type) \
  .load(file_location)

display(s_cg_tweets)

# COMMAND ----------

cg_tweets_full = s_cg_tweets.select("*").toPandas()

# COMMAND ----------

cg_tweets_full.info()

# COMMAND ----------

s_cg_tweets.createOrReplaceTempView("s_cg_tweets")

# COMMAND ----------

# MAGIC %sql select screen_name, text, retweet_count from s_cg_tweets order by retweet_count desc

# COMMAND ----------

# MAGIC %sql CREATE OR REPLACE TEMPORARY VIEW entities AS SELECT entities.hashtags, entities.media, entities.symbols, entities.urls, entities.user_mentions as user_mentions from s_cg_tweets

# COMMAND ----------

# MAGIC %sql CREATE OR REPLACE TEMPORARY VIEW user_mentions AS SELECT user_mentions.id, user_mentions.name, user_mentions.screen_name FROM entities WHERE size(user_mentions) != 0

# COMMAND ----------

# MAGIC %sql select * from user_mentions

# COMMAND ----------

names_df = sql("SELECT row_number() over(order by (select 1)) as number, names from user_mentions lateral view explode(name) as names")
names_df.createOrReplaceTempView("name")

# COMMAND ----------

# MAGIC %sql select * from name

# COMMAND ----------

screen_names_df = sql("SELECT row_number() over(order by (select 1)) as number, screen_names from user_mentions lateral view explode(screen_name) as screen_names")
screen_names_df.createOrReplaceTempView("screen_name")

# COMMAND ----------

# MAGIC %sql select * from screen_name

# COMMAND ----------

df_mentions = sql("SELECT first(names) as name, screen_names as screen_name, count(*) as mention_count from name join screen_name on name.number = screen_name.number group by screen_names order by count(*) desc")
df_mentions.createOrReplaceTempView("mentions")

# COMMAND ----------

# MAGIC %sql select * from mentions

# COMMAND ----------

cg_tweets_full['in_reply_to_status_id'] = cg_tweets_full['in_reply_to_status_id'].astype(pd.Int64Dtype())
cg_tweets_full['in_reply_to_status_id_str'] = cg_tweets_full['in_reply_to_status_id_str'].astype(str)
cg_tweets_full['in_reply_to_user_id'] = cg_tweets_full['in_reply_to_user_id'].astype(pd.Int64Dtype())
cg_tweets_full['in_reply_to_user_id_str'] = cg_tweets_full['in_reply_to_user_id_str'].astype(str)
cg_tweets_full['quoted_status_id'] = cg_tweets_full['quoted_status_id'].astype(pd.Int64Dtype())
cg_tweets_full['quoted_status_id_str'] = cg_tweets_full['quoted_status_id_str'].astype(str)

# COMMAND ----------

rr = sql("SELECT in_reply_to_screen_name as screen_name, count(*) as replies_received FROM s_cg_tweets where in_reply_to_screen_name != 'null' group by in_reply_to_screen_name order by count(*) desc")
rr.createOrReplaceTempView("replies_r")

# COMMAND ----------

# MAGIC %sql CREATE OR REPLACE TEMPORARY VIEW tc AS SELECT first(screen_name) as screen_name, count(*) as tweet_count FROM s_cg_tweets group by user_id order by count(*) desc

# COMMAND ----------

cg_tweets_full[['text_start','text_stop']] = pd.DataFrame(cg_tweets_full.display_text_range.values.tolist(), index=cg_tweets_full.index)

# COMMAND ----------

cg_tweets_full['created_at'] = pd.to_datetime(cg_tweets_full['created_at'],unit='s')

# COMMAND ----------

cg_tweets_full.info()

# COMMAND ----------

cg_tweet = cg_tweets_full[['id','screen_name','created_at','favorite_count','favorited','retweet_count', 'text','text_start','text_stop']]

# COMMAND ----------

print("start date: ", cg_tweet.created_at.min())
print("end date: ", cg_tweet.created_at.max())

# COMMAND ----------

just_the_text = []
for idx in cg_tweet.index:
  tweet = cg_tweet.loc[idx]['text']
  start = cg_tweet.loc[idx]['text_start']
  stop = cg_tweet.loc[idx]['text_stop']
  just_the_text.append(tweet[start:stop])
  
cg_tweet['just_text'] = just_the_text

# COMMAND ----------

cg_tweet = cg_tweet.rename(columns={"text": "full_text"})

# COMMAND ----------

cg_tweet.created_at.hist()

# COMMAND ----------

pd.set_option('display.max_colwidth', -1)
cg_tweet.head(25)

# COMMAND ----------

for tweet in cg_tweet.full_text:
  if (tweet[:2] =='.@'):
    print(tweet)

# COMMAND ----------

tweet_freq_hour = pysqldf("SELECT strftime('%H', created_at) as hour, count(*) as cnt FROM cg_tweet group by hour;")
tweet_freq_month = pysqldf("SELECT strftime('%m', created_at) as month, count(*) as cnt FROM cg_tweet group by month;")
tweet_freq_year = pysqldf("SELECT strftime('%Y', created_at) as year, count(*) as cnt FROM cg_tweet group by year;")

tweet_freq_year

# COMMAND ----------

def descriptive_stats(table_name, table_col):
  print("================================================")
  print(f"Table Name: {table_name}")
  print(f"Table Col: {table_col}")
  print("================================================")
  
  print("")
  
  # Count
  cnt = pysqldf(f"""SELECT "COUNT" as Stat, 
          count({table_col}) as Value 
          FROM {table_name};""")
  
  # Mean
  mean = pysqldf(f"""SELECT "MEAN" as Stat, 
           CAST(AVG({table_col}) as int) as Value
           FROM {table_name}""")
  
  # Median
  median = pysqldf(f"""SELECT "MEDIAN" as Stat,
             {table_col} as Value
             FROM {table_name}
             ORDER BY {table_col}
             LIMIT 1
             OFFSET (SELECT COUNT(*) FROM {table_name}) / 2
             """)
  
  # Mode
  mode = pysqldf(f"""SELECT "MODE" as Stat,
           {table_col} as Value
           FROM {table_name}
           GROUP BY {table_col}
           ORDER BY COUNT(*) DESC
           LIMIT 1
           """)
  
  # Min and Max
  min = pysqldf(f"""SELECT 'MIN' as Stat,
          MIN({table_col}) as Value
          FROM {table_name}""")
  
  max = pysqldf(f"""SELECT 'MAX' as Stat,
          MAX({table_col}) as Value
          FROM {table_name}""")
  
  display(pd.concat([cnt, mean, median, mode, min, max]))
  print("")

# COMMAND ----------

# For 25-50-75-100 quartile range in SQL

# Using CTEs (Temp Tables) create a percentile breakdown (given parameter entered), then select last_value of the percentile breakdown

def get_iqrs(table_name, table_col):
  query = f"""WITH percent_tbl as (
            SELECT {table_col},
            ntile(4) OVER(
            ORDER BY {table_col}
            ) percent
            FROM {table_name}),
            p_25 as (SELECT (percent * 0.25) as pct,
              last_value({table_col})
              over(partition by percent) last_val
              from percent_tbl
              where percent = 1 limit 1),
            p_50 as (SELECT (percent * 0.25) as pct,
              last_value({table_col})
              over(partition by percent) last_val
              from percent_tbl
              where percent = 2 limit 1),
            p_75 as (SELECT (percent * 0.25) as pct,
              last_value({table_col})
              over(partition by percent) last_val
              from percent_tbl
              where percent = 3 limit 1),
            p_100 as (SELECT (percent * 0.25) as pct,
              last_value({table_col})
              over(partition by percent) last_val
              from percent_tbl
              where percent = 4 limit 1)
            SELECT * from p_25 UNION
            SELECT * from p_50 UNION
            SELECT * from p_75 UNION
            SELECT * from p_100;"""
  iqrs = pysqldf(query)
  return display(pd.DataFrame(iqrs))

# COMMAND ----------

descriptive_stats('tweet_freq_year','tweet_freq_year.cnt')
get_iqrs('tweet_freq_year','cnt')
descriptive_stats('tweet_freq_month','tweet_freq_month.cnt')
get_iqrs('tweet_freq_month','cnt')
descriptive_stats('tweet_freq_hour','tweet_freq_hour.cnt')
get_iqrs('tweet_freq_hour','cnt')

# COMMAND ----------

pd.set_option('float_format', '{:f}'.format)
cg_tweet.describe()

# COMMAND ----------

def word_count(str):
  counts = dict()
  words = str.split()
  
  for word in words:
    if word in counts:
      counts[word] += 1
    else:
      counts[word] = 1
      
  return counts

# Takes an arrayof tweets and gives me counts
def tweet_counts(tweet_array,threshold=10):
  unique, counts = np.unique(np.array(tweet_array), return_counts=True)
  count_dict = dict(zip(unique, counts))
  
  return [(k,v) for k, v in count_dict.items() if v > threshold]

# COMMAND ----------

# Table that holds info about each user and the tweet they were part of (e.g. public )

public_tweets = []
for idx in cg_tweet.index:
  tweet = cg_tweet.loc[idx]["full_text"]
  if (tweet[:2]=='.@'):
    # It's a public tweet on purpose
    parts = tweet.split(" ")
    public_tweets.append(parts[0][2:])
    
tweet_counts(public_tweets,5)

public_t_df = pd.DataFrame(tweet_counts(public_tweets,5), columns=['screen_name', 'count']).sort_values('count', ascending = False)
s_public = spark.createDataFrame(public_t_df)
s_public.createOrReplaceTempView("public")

# COMMAND ----------

public_t_df = public_t_df.rename(columns={"count": "public_count"})
s_public = spark.createDataFrame(public_t_df)
s_public.createOrReplaceTempView("public")

# COMMAND ----------

# MAGIC %sql select * from public

# COMMAND ----------

replies = []
for tweet in cg_tweet.full_text:
  if (tweet[:1]=='@'):
    #It's a reply
    parts = tweet.split()
    for i in range(0, len(parts)):
      if (parts[i][0]=='@'):
        replies.append(parts[i])

# COMMAND ----------

tweet_counts(replies, 5)

# COMMAND ----------

replies_df = pd.DataFrame(tweet_counts(replies,5), columns=['who', 'count']).sort_values('count', ascending = False)
replies_df

# COMMAND ----------

rts = []
for tweet in cg_tweet.full_text:
  if (tweet[:4]=='RT @'):
    splits = tweet.split(": ")
    who = splits[0].split(" ")
    rts.append(who[1][1:])
    
tweet_counts(rts,5)

rts_df = pd.DataFrame(tweet_counts(rts,5), columns=['screen_name', 'count']).sort_values('count', ascending = False)
rts_df

# COMMAND ----------

rts_df = rts_df.rename(columns={"count": "retweet_count"})
s_rts = spark.createDataFrame(rts_df)
s_rts.createOrReplaceTempView("rts")

# COMMAND ----------

# MAGIC %sql select * from rts

# COMMAND ----------

user_df=sql("select tc.screen_name, name, tweet_count, retweet_count, public_count, mention_count, replies_received from tc full join public on tc.screen_name = public.screen_name full join replies_r on tc.screen_name = replies_r.screen_name full join rts on tc.screen_name = rts.screen_name full join mentions on tc.screen_name = mentions.screen_name")

# COMMAND ----------

user_df.createOrReplaceTempView("user")
pd_user = user_df.select("*").toPandas()

# COMMAND ----------

pd_user.describe()

# COMMAND ----------

# MAGIC %sql select * from user order by replies_received

# COMMAND ----------

plt.style.use("fivethirtyeight")

# COMMAND ----------

plt.figure(figsize=(12,6))
plt.plot(tweet_freq_hour.hour, tweet_freq_hour.cnt)
plt.title("Count of Congressional Tweets by Hour")
plt.ylabel("Tweet count")
plt.xlabel("Hour")
plt.show()

# COMMAND ----------

plt.figure(figsize=(12,6))
plt.plot(tweet_freq_month.month, tweet_freq_month.cnt)
plt.title("Count of Congressional Tweets by Month")
plt.ylabel("Tweet count")
plt.xlabel("Month")
plt.show()

# COMMAND ----------

plt.figure(figsize=(12,6))
plt.plot(tweet_freq_year.year, tweet_freq_year.cnt)
plt.title("Count of Congressional Tweets by Year")
plt.ylabel("Tweet count")
plt.xlabel("Year")
plt.show()

# COMMAND ----------

pysqldf("select just_text from cg_tweet")

# COMMAND ----------

pysqldf("select replace( (replace( (replace( (replace(lower(just_text),' a ',' ')), ' is ',' ')),' it ',' ')),' an ',' ') from cg_tweet")

# COMMAND ----------

stopwords = [
  'i',
'me',
'my',
'myself',
'we',
'our',
'ours',
'ourselves',
'you',
'your',
'yours',
'yourself',
'yourselves',
'he',
'him',
'his',
'himself',
'she',
'her',
'hers',
'herself',
'it',
'its',
'itself',
'they',
'them',
'their',
'theirs',
'themselves',
'what',
'which',
'who',
'whom',
'this',
'that',
'these',
'those',
'am',
'is',
'are',
'was',
'were',
'be',
'been',
'being',
'have',
'has',
'had',
'having',
'do',
'does',
'did',
'doing',
'a',
'an',
'the',
'and',
'but',
'if',
'or',
'because',
'as',
'until',
'while',
'of',
'at',
'by',
'for',
'with',
'about',
'against',
'between',
'into',
'through',
'during',
'before',
'after',
'above',
'below',
'to',
'from',
'up',
'down',
'in',
'out',
'on',
'off',
'over',
'under',
'again',
'further',
'then',
'once',
'here',
'there',
'when',
'where',
'why',
'how',
'all',
'any',
'both',
'each',
'few',
'more',
'most',
'other',
'some',
'such',
'no',
'nor',
'not',
'only',
'own',
'same',
'so',
'than',
'too',
'very',
's',
't',
'can',
'will',
'just',
'don',
'should',
'now',
  '&amp;',
  '',
  '.',
  'rt',
  '-',
  'w/',
  '&'
]

sw_df = pd.DataFrame(stopwords, columns=['stopword'])

# COMMAND ----------

from collections import Counter

no_sw = cg_tweet.just_text.apply(lambda word: [word.lower() for word in word.split(' ') if word.lower() not in stopwords])

corpus = []
for ea in no_sw:
  corpus += ea
wordDict = Counter(corpus)

[ (k, v) for k, v in sorted(wordDict.items(), key=lambda item: item[1], reverse=True) ][:20]

# COMMAND ----------

# Create separate result sets for each year ...

from wordcloud import WordCloud

cg_2008 = pysqldf("SELECT just_text from cg_tweet where strftime('%Y', created_at) = '2008' ")
cg_2009 = pysqldf("SELECT just_text from cg_tweet where strftime('%Y', created_at) = '2009' ")
cg_2010 = pysqldf("SELECT just_text from cg_tweet where strftime('%Y', created_at) = '2010' ")
cg_2011 = pysqldf("SELECT just_text from cg_tweet where strftime('%Y', created_at) = '2011' ")
cg_2012 = pysqldf("SELECT just_text from cg_tweet where strftime('%Y', created_at) = '2012' ")
cg_2013 = pysqldf("SELECT just_text from cg_tweet where strftime('%Y', created_at) = '2013' ")
cg_2014 = pysqldf("SELECT just_text from cg_tweet where strftime('%Y', created_at) = '2014' ")
cg_2015 = pysqldf("SELECT just_text from cg_tweet where strftime('%Y', created_at) = '2015' ")
cg_2016 = pysqldf("SELECT just_text from cg_tweet where strftime('%Y', created_at) = '2016' ")
cg_2017 = pysqldf("SELECT just_text from cg_tweet where strftime('%Y', created_at) = '2017' ")

def calc_and_display_wc(df_in):
  df = df_in.copy()
  df['text_no_sw'] = df.just_text.apply(lambda word: [word.lower() for word in word.split(' ') if word.lower() not in stopwords] )
  year_corpus = []
  for ea in df.text_no_sw:
    year_corpus += ea
  wordDict = Counter(year_corpus)
  wordcloud = WordCloud().generate_from_frequencies(wordDict)
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  plt.show()

# COMMAND ----------

for year in range(2008, 2018):
  df_name = 'cg_' + str(year)
  print('CG Tweets from ', year)
  calc_and_display_wc(eval(df_name))

# COMMAND ----------

from textblob import TextBlob

cg_tweet['sentiment'] = cg_tweet.just_text.apply(lambda text: TextBlob(text).sentiment[0])

# COMMAND ----------

pysqldf("select * from cg_tweet").head()

# COMMAND ----------

pysqldf("select * from cg_tweet where sentiment >= 0.0 order by sentiment desc")

# COMMAND ----------

pysqldf("select * from cg_tweet where sentiment < 0.0 order by sentiment")

# COMMAND ----------

cg_tweet.sentiment.hist()
plt.title("Sentiment Analysis -- full range of tweets")

# COMMAND ----------

for year in range(2008, 2018):
  df_name = 'cg_' + str(year)
  print('CG Tweet from', year)
  temp_df = eval(df_name)
  temp_df['sentiment'] = temp_df.just_text.apply(lambda text: TextBlob(text).sentiment[0])
  display(temp_df.sentiment.hist())
  plt.show()

# COMMAND ----------

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF

cg_tweet['text_no_sw'] = cg_tweet.just_text.apply(lambda word: " ".join([word.lower() for word in word.split(' ') if word.lower() not in stopwords]))

#Create my bag of words ... and then my tera-doc matrix (X)
bow = CountVectorizer()
X = bow.fit_transform(cg_tweet.text_no_sw)

# COMMAND ----------

# Keep a human readable index to the terms
index2word = np.array(bow.get_feature_names())

# Use non-negative matrix factorization
nmf = NMF(n_components=9, solver="mu") # Multiplicative Update most widely used
W = nmf.fit_transform(X)
H = nmf.components_

#Print out the resultant topic clusters
for i, topic in enumerate(H):
  print("Topic Cluster {}: {}".format(i + 1, ",".join([str(term) for term in index2word[topic.argsort()[-15:]]])))

# COMMAND ----------

# File location and type
file_location = "/user/hive/warehouse/tweet_user"
file_type = "parquet"

# The applied options are for CSV files. For other file types, these will be ignored.
user = spark.read.format(file_type) \
  .load(file_location)

# COMMAND ----------

user.createOrReplaceTempView("user")

# COMMAND ----------

# MAGIC %sql select * from user where screen_name != 'null' order by retweet_count desc
