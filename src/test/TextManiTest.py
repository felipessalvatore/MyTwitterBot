import unittest
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils import run_test
from text_processing.functions import read_line_eos, clean_text
from text_processing.functions import file_len, text_cut, clean_and_cut
from text_processing.Vocab import Vocab


class TextManiTest(unittest.TestCase):
    """
    Class that test the functions from the
    text_mani module
    """
    @classmethod
    def setUpClass(cls):
        data_path = os.path.join(parentdir, 'data')
        cls.text_path_toy = os.path.join(data_path, 'toy.txt')
        cls.text_path_tweets = os.path.join(data_path, 'toy_tweets.txt')
        cls.clean_txt_path = cls.text_path_tweets[:-4] + "CLEAN.txt"
        cls.train_path = cls.text_path_tweets[:-4] + "TRAIN.txt"
        cls.valid_path = cls.text_path_tweets[:-4] + "VALID.txt"
        cls.test_path = cls.text_path_tweets[:-4] + "TEST.txt"
        cls.train_path2 = cls.text_path_tweets[:-4] + "CLEANTRAIN.txt"
        cls.valid_path2 = cls.text_path_tweets[:-4] + "CLEANVALID.txt"
        cls.test_path2 = cls.text_path_tweets[:-4] + "CLEANTEST.txt"
        cls.first = 'Am I to become profligate as if I were a blonde'
        cls.first = cls.first + ' Or religious as if I were French'
        cls.last = "Meditations in an Emergency by Frank O'Hara"

    @classmethod
    def tearDown(cls):
        if os.path.exists(cls.clean_txt_path):
            os.remove(cls.clean_txt_path)
        if os.path.exists(cls.train_path):
            os.remove(cls.train_path)
        if os.path.exists(cls.valid_path):
            os.remove(cls.valid_path)
        if os.path.exists(cls.test_path):
            os.remove(cls.test_path)
        if os.path.exists(cls.train_path2):
            os.remove(cls.train_path2)
        if os.path.exists(cls.valid_path2):
            os.remove(cls.valid_path2)
        if os.path.exists(cls.test_path2):
            os.remove(cls.test_path2)

    def test_len_text(self):
        """
        Testing the function file_len
        """
        result = file_len(TextManiTest.text_path_tweets)
        self.assertEqual(result, 10,
                         msg="{}".format(result))

    def test_cutting(self):
        """
        Test if the cutting fuction
        is separeting the text in train (4\5),
        valid (1\5), test (1\5).
        """
        text_cut(TextManiTest.text_path_tweets)
        result_train = file_len(TextManiTest.train_path)
        result_valid = file_len(TextManiTest.valid_path)
        result_test = file_len(TextManiTest.test_path)
        self.assertEqual(result_train,
                         8,
                         msg="{}".format(result_train))
        self.assertEqual(result_valid,
                         1,
                         msg="{}".format(result_valid))
        self.assertEqual(result_test,
                         1,
                         msg="{}".format(result_test))

    def test_cleaning(self):
        """
        Testing the function that removes links, numbers,
        double spaces and emogis from a txt file
        """
        clean_tweet1 = "i have a repository with N folders at ok \n"
        clean_tweet2 = "today is the best day #tensorflow \n"
        clean_text(TextManiTest.text_path_tweets)
        new_text = []
        for line in open(TextManiTest.clean_txt_path):
            new_text.append(line)
        self.assertEqual(new_text[0],
                         clean_tweet1,
                         msg="{}".format(new_text[0]))
        self.assertEqual(new_text[1],
                         clean_tweet2,
                         msg="{}".format(new_text[1]))

    def test_clean_and_cut(self):
        """
        Testing the clean_and_cut function
        """
        train_path, valid_path, test_path = clean_and_cut(TextManiTest.text_path_tweets)
        train_path, valid_path, test_path = clean_and_cut(TextManiTest.text_path_tweets)
        clean_tweet1 = "i have a repository with N folders at ok \n"
        clean_tweet2 = "today is the best day #tensorflow "
        result_train = file_len(train_path)
        result_valid = file_len(valid_path)
        result_test = file_len(test_path)
        self.assertEqual(result_train,
                         8,
                         msg="{}".format(result_train))
        self.assertEqual(result_valid,
                         1,
                         msg="{}".format(result_valid))
        self.assertEqual(result_test,
                         1,
                         msg="{}".format(result_test))
        for line in open(valid_path):
            result = line
        self.assertEqual(result,
                         clean_tweet1,
                         msg="result = {}".format(result))
        for line in open(test_path):
            result = line
        self.assertEqual(result,
                         clean_tweet2,
                         msg="result = {}".format(result))

    def test_enco_deco(self):
        """
        Testing the encoding and decoding of the class Vocab
        """
        vocab = Vocab()
        path = TextManiTest.text_path_toy
        sentence = []
        vocab.read_words(read_line_eos(path))
        encode = [vocab.encode(word) for word in read_line_eos(path)]
        for word in [vocab.decode(index) for index in encode]:
            if word != "<eos>":
                sentence.append(word)
            else:
                break
        result = ' '.join(sentence)
        self.assertEqual(result,
                         TextManiTest.first,
                         msg="result = {}".format(result))

    def test_eos(self):
        """
        Testing if the function read_line_eos
        is adding the str <eof> at the end of each line

        """
        vocab = Vocab()
        path = TextManiTest.text_path_toy
        sentence = []
        count = 0
        vocab.read_words(read_line_eos(path))
        encode = [vocab.encode(word) for word in read_line_eos(path)]
        for word in [vocab.decode(index) for index in encode]:
            if word != "<eos>":
                if count == 13:
                    sentence.append(word)
            else:
                count += 1
        result = ' '.join(sentence)
        self.assertEqual(result,
                         TextManiTest.last,
                         msg="result = {}".format(result))


if __name__ == "__main__":
    run_test(TextManiTest,
             "\n=== Running test for the text manipulation functions ===\n")
