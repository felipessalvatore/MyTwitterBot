import numpy as np
import os
import sys
import inspect
from textblob import TextBlob

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from text_processing.Vocab import Vocab
from text_processing.functions import clean_and_cut


class DataHolder():
    """
    Class that preprocess all the text data. It also
    stores a list of noums of the original text.
    We use this list to substititute the tokens "unk_token"
    from the text. The textblob library adds some blanck space.
    That is why we used two unk variables: "unk_token" is the
    original token, "blob_unk_token" is the one for the textblob lib.
    The default text is the ptb dataset that is stored in the folder "data".
    So if you pass a folder as the "text_path" it will search the
    ptb dataset. Otherwise it will prepocess the txt in "text_path".

    :type text_path: str
    :type debug: boolean
    :type max_noums: int
    :type unk_token: str
    """
    def __init__(self,
                 text_path,
                 debug=False,
                 max_noums=100,
                 unk_token='<unk>'):
        self.unk_token = unk_token
        self.blob_unk_token = '< unk >'
        self.all_noums = set([])
        self.max_noums = max_noums
        if os.path.isdir(text_path):
            self.path_train = os.path.join(text_path, 'ptb.train.txt')
            self.path_valid = os.path.join(text_path, 'ptb.valid.txt')
            self.path_test = os.path.join(text_path, 'ptb.test.txt')
            assert os.path.exists(self.path_train), "wrong folder!"
            assert os.path.exists(self.path_valid), "wrong folder!"
            assert os.path.exists(self.path_test), "wrong folder!"
        else:
            path_train, path_valid, path_test = clean_and_cut(text_path)
            self.path_train = path_train
            self.path_valid = path_valid
            self.path_test = path_test
        self.load_data(debug)
        self.all_noums = [noum for noum in list(self.all_noums)
                          if noum.find(self.blob_unk_token) == -1]

    def read_line_eos_noums(self,
                            path):
        """
        Generator.
        Similar as the function read_line_eos from
        the text_mani module. The only diference here
        is that we keep track of all the noums.

        :type path: str
        """
        for line in open(path):
            if len(list(self.all_noums)) <= self.max_noums:
                blob = TextBlob(line)
                noums = set(blob.noun_phrases)
                self.all_noums = self.all_noums.union(noums)
            for word in line.split():
                yield word
            yield '<eos>'

    def load_data(self, debug):
        """
        Loads starter word-vectors and train/dev/test data.

        :type debug: boolean
        """
        self.vocab = Vocab()
        self.vocab.read_words(self.read_line_eos_noums(self.path_train))
        self.encoded_train = np.array(
            [self.vocab.encode(word)
             for word in self.read_line_eos_noums(self.path_train)],
            dtype=np.int32)
        self.encoded_valid = np.array(
            [self.vocab.encode(word)
             for word in self.read_line_eos_noums(self.path_valid)],
            dtype=np.int32)
        self.encoded_test = np.array(
            [self.vocab.encode(word)
             for word in self.read_line_eos_noums(self.path_test)],
            dtype=np.int32)
        if debug:
            num_debug = 1024
            self.encoded_train = self.encoded_train[:num_debug]
            self.encoded_valid = self.encoded_valid[:num_debug]
            self.encoded_test = self.encoded_test[:num_debug]
