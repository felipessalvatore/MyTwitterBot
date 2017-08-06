import os
import argparse
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentparentdir)

from tftools.Config import Config
from tftools.DataHolder import DataHolder
from tftools.RNNLanguageModel import RNNLanguageModel
from tftools.train_functions import train_model


SafatleCorpus = os.path.join(parentparentdir, "data", "SafatleCorpus.txt")
my_config = Config()
my_dataholder = DataHolder(text_path=SafatleCorpus)
model = RNNLanguageModel(config=my_config,
                         dataholder=my_dataholder)
train_model(model)
