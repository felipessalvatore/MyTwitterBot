import unittest
import os
import sys
import inspect
import shutil
import tensorflow as tf
from copy import deepcopy

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils import run_test
from tftools.Config import Config
from tftools.DataHolder import DataHolder
from tftools.RNNLanguageModel import RNNLanguageModel
from tftools.train_functions import run_epoch
from tftools.generate_functions import generate_text


class GenerateFunctionsTest(unittest.TestCase):
    """
    Class that test the functions from the
    generate_functions module
    """
    @classmethod
    def setUpClass(cls):
        cls.config = Config(max_epochs=1)
        data_path = os.path.join(parentdir, "data")
        cls.data = DataHolder(text_path=data_path)
        cls.model = RNNLanguageModel(cls.config, cls.data)
        cls.gen_config = deepcopy(cls.config)
        cls.gen_config.batch_size = cls.gen_config.num_steps = 1
        OneTwentyFive = int(len(cls.model.encoded_train) / 25)
        cls.toy_train = cls.model.encoded_train[0: OneTwentyFive]

    @classmethod
    def tearDown(cls):
        check_path = os.path.join(currentdir, "checkpoints")
        logs_path = os.path.join(currentdir, "logs")
        if os.path.exists(check_path):
            shutil.rmtree(check_path)
        if os.path.exists(logs_path):
            shutil.rmtree(logs_path)

    def test_generate_text(self):
        """
        Testing if the generate_text function generates a texts
        """
        model = GenerateFunctionsTest.model
        gen_config = GenerateFunctionsTest.gen_config
        toy_train = GenerateFunctionsTest.toy_train
        with tf.Session(graph=model.graph) as sess:
                tf.global_variables_initializer().run()
                run_epoch(model, sess, toy_train, model.train_op)
                model.saver.save(sess, model.save_path)
        gen_model = RNNLanguageModel(gen_config,
                                     GenerateFunctionsTest.data)
        with tf.Session(graph=gen_model.graph) as sess:
                gen_model.saver.restore(sess, gen_model.save_path)
                result = ' '.join(generate_text(sess,
                                                gen_model,
                                                gen_config,
                                                "i am",
                                                stop_tokens=['<eos>']))
        self.assertEqual(type(result),
                         str,
                         msg="not str\n type(result) ={}".format(type(result)))
        self.assertTrue(len(result) > 1,
                        msg="len(result) = {}".format(len(result)))
        test1 = result[-5:] == '<eos>'
        test2 = len(result.split()) == 102
        self.assertTrue(test1 or test2,
                        msg="result = {}".format(result))


if __name__ == "__main__":
    run_test(GenerateFunctionsTest,
             "\n=== Running test for the generate functions ===\n")
