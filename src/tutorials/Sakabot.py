import unittest
import os
import sys
import inspect
import shutil
import tensorflow as tf

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from tftools.Config import Config
from twitter.TweetGenerator import TweetGenerator

if __name__ == "__main__":
    config = Config()
    data_path = os.path.join(parentdir, "data", "SakaCorpus.txt")
    tg = TweetGenerator(text_path=data_path,
                        config=config,
                        train=True)
    tweet_list = tg.generate_tweet_list(50, "O Capitalismo é")
    for tweet in tweet_list:
        print(tweet)
