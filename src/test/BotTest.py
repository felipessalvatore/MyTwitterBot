import unittest
import os
import sys
import inspect
import shutil
import pandas as pd

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils import run_test
from agent.Bot import Bot


class BotTest(unittest.TestCase):
    """
    Class that test the twitter Bot.
    """
    @classmethod
    def setUpClass(cls):
        cls.csv_path = os.path.join(currentdir, "twitter_log", "stats.csv")
        cls.data_path = os.path.join(parentdir, "data")

    @classmethod
    def tearDown(cls):
        check_path = os.path.join(currentdir, "checkpoints")
        logs_path = os.path.join(currentdir, "twitter_log")
        if os.path.exists(check_path):
            shutil.rmtree(check_path)
        if os.path.exists(logs_path):
            shutil.rmtree(logs_path)

    def test_log(self):
        """
        Everytime we create one bot he saves the
        twitter status in a csv file. This function tests
        if he is saving the correct information.
        """
        Bot(corpus=BotTest.data_path)
        self.assertTrue(os.path.exists(BotTest.csv_path),
                        msg="Not writing csv for the first time")
        Bot(corpus=cls.data_path)
        df = pd.read_csv(BotTest.csv_path)
        self.assertEqual(df.shape, (2, 4),
                         msg="Wrong Shape\n {}".format(df))


if __name__ == "__main__":
    key_path = os.path.join(parentdir, "agent", "key.py")
    if os.path.exists(key_path):
        run_test(BotTest,
                 "\n=== Running test for the Twitter Bot ===\n")
    else:
        print("No file in the path \n {}".format(key_path))
