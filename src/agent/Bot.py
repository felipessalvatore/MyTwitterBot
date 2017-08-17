import tweepy
import pandas as pd
import time
import numpy as np

try:
    from key import ConsumerKey, ConsumerSecret
    from key import AccessToken, AccessTokenSecret
except ImportError:
    from agent.key import ConsumerKey, ConsumerSecret
    from agent.key import AccessToken, AccessTokenSecret

import os
import sys
import inspect
from requests_oauthlib import OAuth1Session
import json

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils import get_real_friends, get_date, get_date_and_time
from twitter.TweetGenerator import TweetGenerator
from twitter.functions import TweetValid
from text_processing.functions import file_len


class Bot():
    """
    The autonomous agent behind the twitter account.
    This class assumes that you have the file "key.py"
    in the folder "agent".
    In "key.py" I assume you have the variables:
    "ConsumerKey" , "ConsumerSecret", "AccessToken"
    and "AccessTokenSecret". For more info on how to get
    the value of these variables go watch this video on
    youtube https://www.youtube.com/watch?v=M7MqML2ZVOY

    :type corpus: str
    :type friends: list of str
    :type commentary: srt
    :type black_list: list
    :type local: str
    :type hashtag_search: None or list
    """
    def __init__(self,
                 corpus,
                 friends=[],
                 commentary="None",
                 black_list=[],
                 local="world",
                 hashtag_search=None):
        self.black_list = black_list
        self.local = local
        self.friends = friends
        self.corpus = corpus
        auth = tweepy.OAuthHandler(ConsumerKey, ConsumerSecret)
        auth.set_access_token(AccessToken, AccessTokenSecret)
        self.api = tweepy.API(auth)
        entry = [("Date", [get_date()]),
                 ("Followers", [len(self.api.followers_ids())]),
                 ("Following", [len(self.api.friends_ids())]),
                 ("Commentary", [commentary])]
        self.df = pd.DataFrame.from_items(entry)
        self.log()
        if hashtag_search is None:
            self.hashtag_search = self.get_trends(self.local)
        else:
            self.hashtag_search = hashtag_search + self.get_trends(self.local)

    def clear_follow(self,
                     Realfriends=get_real_friends()):
        """
        Method to remove all the people that the bot followers
        that are not in the list "Realfriends"

        :type Realfriends: list of int
        """
        friends = self.api.friends_ids()
        for friend in friends:
            if friend not in Realfriends:
                self.api.destroy_friendship(friend)

    def log(self):
        """
        Method to save the twitter status on a csv for
        future reference.
        """
        log_folder = os.path.join(os.getcwd(), "twitter_log")
        csv_name = os.path.join(log_folder, "stats.csv")
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        try:
            old_df = pd.read_csv(csv_name)
            new_df = old_df.append(self.df, ignore_index=True)
            new_df.to_csv(csv_name, index=False)
        except OSError:
            self.df.to_csv(csv_name, index=False)

    def get_local_identifier(self):
        """
        Method to get dict local: identifier.
        the identifier is of type WOEID (Where On Earth IDentifier).

        :rtype: dict
        """
        WOEID = {"world": "1",
                 "EUA": "23424977",
                 "Brazil": "23424768"}
        return WOEID

    def get_trends(self, local):
        """
        Method to get the trending hashtags.

        :type local: str
        :rtype: list of str
        """
        session_string = "https://api.twitter.com/1.1/trends/place.json?id="
        local_id = self.get_local_identifier()[local]
        session_string += local_id
        session = OAuth1Session(ConsumerKey,
                                ConsumerSecret,
                                AccessToken,
                                AccessTokenSecret)
        response = session.get(session_string)
        if response.__dict__['status_code'] == 200:
            local_trends = json.loads(response.text)[0]["trends"]
            hashtags = [trend["name"]
                        for trend in local_trends if trend["name"][0] == '#']
        else:
            hashtags = []
        return hashtags

    def curator_writer(self,
                       num_tweets,
                       show_tweets=10,
                       num_hashtags=5):
        """
        Method to write "num_tweets" tweets. Here I use a loop
        to get an input to the user to choose one tweet.
        At the end of the loop the method write a txt file with
        all the tweets. We use the trending hashtags and the bot's
        frieds to compose the tweet.

        :type num_tweets: int
        :type num_hashtags: int
        :rtype: str
        """
        saved_tweets = []
        tg = TweetGenerator(text_path=self.corpus,
                            black_list=self.black_list,
                            train=False)
        while len(saved_tweets) < num_tweets:
            print(('=-=' * 5))
            print("You have {} saved tweets so far.".format(len(saved_tweets)))
            print("Type the beginning of a tweet")
            print(('=-=' * 5))
            first_part = input('> ')
            if not TweetValid(first_part):
                first_part = '<eos>'
                print("Too long!!\nstarting text = <eos>")
            hashtags = self.get_trends(self.local)
            hashtags_and_friends = self.friends + hashtags
            h_and_f_size = len(hashtags_and_friends)
            if h_and_f_size < num_hashtags:
                num_hashtags = max(len(hashtags_and_friends) - 1, 1)
                print("Picking only {} hashtags".format(num_hashtags))
            if h_and_f_size > 0:
                choice = np.random.choice(h_and_f_size, num_hashtags)
                my_hashtags = [hashtags_and_friends[i] for i in choice]
            else:
                my_hashtags = []
            tweets = tg.generate_tweet_list(number_of_tweets=show_tweets,
                                            starting_text=first_part,
                                            hashtag_list=my_hashtags)
            for i, tweet in enumerate(tweets):
                print("{0}) {1}".format(i, tweet))
            user_choice = -1
            number_of_tweets = len(tweets)
            while True:
                print(('=-=' * 5))
                print("Choose one tweet!")
                print("Type a number from 0 to {}".format(number_of_tweets - 1))
                print("Or type -99 to generate other tweets")
                print(('=-=' * 5))
                user_choice = input('> ')
                try:
                    user_choice = int(user_choice)
                except ValueError:
                    print("Oops! That was no valid number.")
                if user_choice == -99 or user_choice in range(number_of_tweets):
                    break
            if user_choice >= 0:
                saved_tweets.append(tweets[user_choice])
        draft_folder = os.path.join(os.getcwd(), "twitter_draft")
        filename = os.path.join(draft_folder, get_date_and_time() + ".txt")
        if not os.path.exists(draft_folder):
            os.makedirs(draft_folder)
        with open(filename, "w") as f:
            for tweet in saved_tweets:
                f.write(tweet + "\n")
        return filename

    def post_from_txt(self,
                      text_path,
                      minutes_paused=2,
                      num_tweets=51):
        """
        Method to post all the tweets from the txt in "text_path".
        Each tweet is posted and after that the bot starts to
        liking tweets that have the same hasthags as the ones in the list
        self.hashtag_search, the bot also retweet the theets and follow the
        user. After that it pause for "minutes_paused" minutes
        (default is 2 minutes).

        :type text_path: str
        :type minutes_paused: int
        :type num_tweets: int
        """
        seconds_pause = minutes_paused * 60
        num_tweets = file_len(text_path)
        with open(text_path) as file:
            for i, tweet in enumerate(file):
                if TweetValid(tweet):
                    print("Posting {0} from {1}".format(i, num_tweets))
                    self.api.update_status(tweet)
                    choice = np.random.choice(len(self.hashtag_search), 1)[0]
                    current_hashtag = self.hashtag_search[choice]
                    print("\ncurrent hashtag is {}".format(current_hashtag))
                    count = 0
                    for tweet in tweepy.Cursor(self.api.search,
                                               q=current_hashtag).items():
                        print("\ncount = {}".format(count))
                        if count < num_tweets:
                            try:
                                # Favorite the tweet
                                tweet.favorite()
                                print('Favorited the tweet')
                                # Follow the user who tweeted
                                tweet.user.follow()
                                print('Followed the user')
                                if count % 25 == 0:
                                    tweet.retweet()
                                    print('Retweeted the tweet')
                                print("\nWaiting {} minutes".format(minutes_paused))
                                time.sleep(seconds_pause)
                                count += 1

                            except tweepy.TweepError as e:
                                print(e.reason)

                            except StopIteration:
                                print("No more tweets for the hashtag = {}".format(current_hashtag))
                                break
                        else:
                            break

    def write(self,
              num_tweets,
              first_part='<eos>',
              num_hashtags=5,
              minutes_pause=60,
              publish=True):
        """
        Method to write "num_tweets" tweets, using the string
        "first part" as the begining of the tweet and
        using "num_hashtags"  hashtags
        Each tweet is posted after a pause of
        "minutes_pause" minutes (default is one hour).

        :type num_tweets: int
        :type num_hashtags: int
        :type minutes_pause: int
        :type publish: boolean
        """
        seconds_pause = minutes_pause * 60
        tg = TweetGenerator(text_path=self.corpus,
                            black_list=self.black_list,
                            train=False)
        for i in range(num_tweets):
            trends = self.api.trends_place(1)[0]['trends']
            TrendsNames = [trend['name'] for trend in trends]
            hashtags = [words for words in TrendsNames if words[0] == "#"]
            if len(hashtags) < num_hashtags:
                num_hashtags = max(len(hashtags) - 1, 1)
                print("Picking only {} hashtags".format(num_hashtags))
            choice = np.random.choice(len(hashtags), num_hashtags)
            my_hashtags = [hashtags[i] for i in choice]
            tweet = tg.generate_tweet_list(starting_text=first_part,
                                           hashtag_list=my_hashtags)[0]
            print("\nThe {} tweet is:\n".format(i), tweet)
            if publish:
                self.api.update_status(tweet)
                print("Waiting {} minutes".format(minutes_pause))
                time.sleep(seconds_pause)
