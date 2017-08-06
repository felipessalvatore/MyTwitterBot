import unittest
import os
import sys
import inspect
import shutil
import tensorflow as tf

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils import run_test
from tftools.Config import Config
from tftools.DataHolder import DataHolder
from tftools.RNNLanguageModel import RNNLanguageModel
from tftools.train_functions import run_epoch


class RNNTest(unittest.TestCase):
    """
    Class that test the RNN model
    """
    @classmethod
    def setUpClass(cls):
        cls.config = Config(max_epochs=1)
        data_path = os.path.join(parentdir, "data")
        cls.data = DataHolder(text_path=data_path)
        cls.model = RNNLanguageModel(cls.config, cls.data)
        OneFourth = int(len(cls.model.encoded_valid) / 4)
        OneTwentyFive = int(len(cls.model.encoded_train) / 25)
        cls.toy_valid = cls.model.encoded_valid[0: OneFourth]
        cls.toy_train = cls.model.encoded_train[0: OneTwentyFive]

    @classmethod
    def tearDown(cls):
        check_path = os.path.join(currentdir, "checkpoints")
        logs_path = os.path.join(currentdir, "logs")
        if os.path.exists(check_path):
            shutil.rmtree(check_path)
        if os.path.exists(logs_path):
            shutil.rmtree(logs_path)

    def test_otimization(self):
        """
        Testing if the perpexity on the valid data
        is going down
        """
        model = RNNTest.model
        toy_valid = RNNTest.toy_valid
        toy_train = RNNTest.toy_train
        with tf.Session(graph=model.graph) as sess:
                tf.global_variables_initializer().run()
                before_training = run_epoch(model, sess, toy_valid)
                run_epoch(model, sess, toy_train, model.train_op)
                after_traing = run_epoch(model, sess, toy_valid)
        self.assertTrue(before_training > after_traing,
                        msg="before = {0}\nafter = {1}".format(before_training,
                                                               after_traing))


if __name__ == "__main__":
    run_test(RNNTest,
             "\n=== Running test for the RNN model ===\n")
