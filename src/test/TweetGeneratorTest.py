import unittest
import os
import sys
import inspect
import shutil

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils import run_test
from tftools.Config import Config
from twitter.TweetGenerator import TweetGenerator
from twitter.functions import TweetValid


class TweetGeneratorTest(unittest.TestCase):
    """
    Class that test generation of tweets from the class
    TweetGenerator.
    """
    @classmethod
    def setUpClass(cls):
        cls.config = Config(max_epochs=1, batch_size=30)
        cls.data_path = os.path.join(parentdir, "data")

    @classmethod
    def tearDown(cls):
        check_path = os.path.join(currentdir, "checkpoints")
        logs_path = os.path.join(currentdir, "logs")
        if os.path.exists(check_path):
            shutil.rmtree(check_path)
        if os.path.exists(logs_path):
            shutil.rmtree(logs_path)

    def test_tweet_size(self):
        """
        Function to test if all the sentences that are
        being generated are valid tweets
        """
        tg = TweetGenerator(text_path=TweetGeneratorTest.data_path,
                            config=TweetGeneratorTest.config,
                            train=True,
                            debug=True)
        tweet_list = tg.generate_tweet_list(50, "i am")
        debug = [(TweetValid(tweet), tweet, len(tweet))
                 for tweet in tweet_list]
        result = all([triple[0] for triple in debug])
        self.assertTrue(result, msg="\nAll tweets\n {}".format(debug))

    def test_tweet_size_hashtags(self):
        """
        Function to test if all the sentences that are
        being generated are valid tweets. Now using some
        hastags.
        """
        tg = TweetGenerator(text_path=TweetGeneratorTest.data_path,
                            config=TweetGeneratorTest.config,
                            train=True,
                            debug=True)
        hastags = ["#AI", "#tensorflow"]
        tweet_list = tg.generate_tweet_list(50, "i am", hashtag_list=hastags)
        debug = [(TweetValid(tweet), tweet, len(tweet))
                 for tweet in tweet_list]
        result = all([triple[0] for triple in debug])
        self.assertTrue(result, msg="\nAll tweets\n {}".format(debug))

    def test_tweet_hashtags_content(self):
        """
        Function to test if all the tweets have the
        hashtags from the hastag list
        """
        tg = TweetGenerator(text_path=TweetGeneratorTest.data_path,
                            config=TweetGeneratorTest.config,
                            train=True,
                            debug=True)
        hastags = ["#AI", "#tensorflow"]
        tweet_list = tg.generate_tweet_list(50, "i am", hashtag_list=hastags)
        result = True
        debug = "NoProblemo"
        for tweet in tweet_list:
            condition1 = tweet.find("#AI") != -1
            condition2 = tweet.find("#tensorflow") != -1
            if not (condition1 and condition2):
                debug = tweet
                result = False
                break
        self.assertTrue(result, msg="\nProblematic tweet = {}".format(debug))


if __name__ == "__main__":
    run_test(TweetGeneratorTest,
             "\n=== Running test for tweet generation ===\n")
