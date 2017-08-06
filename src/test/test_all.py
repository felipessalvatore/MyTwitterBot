#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import inspect
from RNNTest import RNNTest
from GenerateFunctionsTest import GenerateFunctionsTest
from TextManiTest import TextManiTest
from TweetGeneratorTest import TweetGeneratorTest

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

user_has_key = True
key_path = os.path.join(parentdir, "agent", "key.py")
try:
    from BotTest import BotTest
except ImportError:
    assert not os.path.exists(key_path)
    user_has_key = False

sys.path.insert(0, parentdir)

from utils import run_test


def main():
    run_test(TextManiTest,
             "\n=== Running test for the text manipulation functions ===\n")
    run_test(RNNTest,
             "\n=== Running test for the RNN model ===\n")
    run_test(GenerateFunctionsTest,
             "\n=== Running test for the generate functions ===\n")
    run_test(TweetGeneratorTest,
             "\n=== Running test for tweet generation ===\n")
    if user_has_key:
        run_test(BotTest,
                 "\n=== Running test for the Twitter Bot ===\n")
    else:
        print("The test for the Twitter Bot was not executed")


if __name__ == "__main__":
    main()
