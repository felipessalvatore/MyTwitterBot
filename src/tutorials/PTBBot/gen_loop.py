import os
import argparse
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentparentdir)

from tftools.Config import Config
from tftools.generate_functions import generate_loop

PTBCorpus = os.path.join(parentparentdir, "data")
my_config = Config()
generate_loop(my_config, text_path=PTBCorpus)
