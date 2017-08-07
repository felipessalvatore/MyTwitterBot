import os
import argparse
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentparentdir)

from agent.Bot import Bot

parser = argparse.ArgumentParser()

parser.add_argument('text_path',
                    type=str, help='path to text')

parser.add_argument("-m",
                    "--minutes",
                    type=int,
                    default=60,
                    help="minutes to wait between posting (default=60)")

user_args = parser.parse_args()

TrumpCorpus = os.path.join(parentparentdir, "data", "TrumpTweets.txt")
my_bot = Bot(corpus=TrumpCorpus, commentary="TrumpBot")
path = my_bot.curator_writer(num_tweets=user_args.tweets,
                             show_tweets=user_args.show,
                             num_hashtags=user_args.hashtags)

print("\n file can be found in {}".format(path))
