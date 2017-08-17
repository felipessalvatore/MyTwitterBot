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
                    default=2,
                    help="minutes to wait between liking (default=2)")

user_args = parser.parse_args()

SakaCorpus = os.path.join(parentparentdir, "data", "SakaCorpus.txt")
# SakaHastag = ["#foratemer",
#               "#foradoria",
#               '#DemocraciaJá',
#               '#ForaTemer',
#               "#foraDória",
#               '#DiarioDoMundo']
my_bot = Bot(corpus=SakaCorpus,
             commentary="SakaBot + local=Brazil + like_retweet_follow",
             local="Brazil",
             hashtag_search=None)
my_bot.post_from_txt(text_path=user_args.text_path,
                     minutes_paused=user_args.minutes)
