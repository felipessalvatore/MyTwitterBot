import collections
import os
import shutil
import tensorflow as tf
from collections import defaultdict
import numpy as np
import re
import sys


class Vocab(object):
    """
    Class to process one text file
    """
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.unknown = '<unk>'
        self.add_word(self.unknown, count=0)

    def add_word(self, word, count=1):
        """
        Add a new word to the vocab dict, giving
        a new index to this word. It also keep track
        of the word frequency.

        :type word: str
        :type count: int
        """
        if word not in self.word2index:
            index = len(self.word2index)
            self.word2index[word] = index
            self.index2word[index] = word
        self.word_freq[word] += count

    def read_words(self, words):
        """
        Add every word from the list of words "words".
        It prints also the vocabulary size and the token count

        :type words: list of str
        """

        for word in words:
            self.add_word(word)
        self.total_words = sum(self.word_freq.values())
        uniques = self.__len__()
        print('{} total tokens with {} uniques'.format(self.total_words,
                                                       uniques))

    def encode(self, word):
        """
        Translation: word to index
        This function assumes that the vocabulary is the one
        learned by the function "read_words". Every word
        that do not appear in the text that was the argument
        of the funtion "read_words" will be regarded as '<unk>'.

        :type word: str
        :rtype: int
        """
        if word not in self.word2index:
            word = self.unknown
        return self.word2index[word]

    def decode(self, index):
        """
        Translation: index to words

        :type index : int
        :rtype: str
        """
        return self.index2word[index]

    def __len__(self):
        """
        Return the number of unique words in the text

        :rtype: int
        """
        return len(self.word_freq)


def read_line_eos(path):
    """
    Generator.
    Read each line of the text in "path" and
    at the end of each line insert the string "<eos>"
    to mark the end of sentence.

    :type path: str
    """
    for line in open(path):
        for word in line.split():
            yield word
        yield '<eos>'


def ptb_iterator(raw_data, batch_size, num_steps):
    """
    Generator.
    "raw_data" is an array of indexes -- shape = (#tokens,).
    Each index correspond to a word. So we can regard "raw_data"
    as the raw text.
    We will transform the text in an array "data".
    Each row of this array is a sequence of words
    with size "batch_size". So this array is of shape
    (batch_size, len(raw_data) // batch_size)

    for each iteration in range(epoch_size):
      - x and y are arrays of shape (batch_size,num_steps).

      - x[i] and y[i] are lines of the text of size "num_steps"

      - y[i][j] is the immediate following word of x[i][j]

      - x[i+1][j] is the point in the text ahead of x[i][j] by batch_len -1
        tokens(<eof> included)


    :type raw_data: list or np array
    :type batch_size: int
    :type num_steps: int
    """
    raw_data = np.array(raw_data, dtype=np.int32)
    batch_len = len(raw_data) // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
    epoch_size = (batch_len - 1) // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)


def file_len(file_path):
    """
    Function that counts the number of lines
    of a txt file.

    :type file_path: str
    :rtype: int
    """
    with open(file_path) as f:
        for i, _ in enumerate(f):
            pass
    return i + 1


def clean_text(path):
    """
    Function that remove every link, number and punctiation
    of a txt file, and create a new txt such that every link is replace
    with the string  "LINK", every @Somebody is replace with 'PERSON',
    every number with "N", every emoji with "EMOJI". Multiple
    spaces are also eliminated. It also puts every
    word in the lower case format.

    :type path: str
    """
    new_path = path[:-4] + "CLEAN.txt"
    url = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    nums = re.compile(r'[+-]?\d+(?:\.\d+)?')
    punct = re.compile(r'[.?\-",!;–…]+')
    friends = re.compile(r'@[A-Za-z0-9]+')
    spaces = re.compile(' +')
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    with open(new_path, "w") as f:
        for line in open(path):
            line = line.lower()
            new_line = url.sub("LINK", line)
            new_line = friends.sub('PERSON', new_line)
            new_line = nums.sub("N", new_line)
            new_line = punct.sub(" ", new_line)
            new_line = emoji_pattern.sub("EMOJI", new_line)
            new_line = spaces.sub(" ", new_line)
            f.write(new_line)


def text_cut(file_path, verbose=False):
    """
    Given the txt file in "file_path" this function
    separetes four fifths of the lines and write them in the file
    "file_pathTRAIN.txt" (that is why we use the
    variable ff).
    The rest of the text is divide into valid and test txts.

    :type file_path: str
    :type verbose: boolean
    """
    tlen = file_len(file_path)
    ff = int((4 / 5) * tlen)
    rest = tlen - ff
    rest = int(rest / 2)
    train_path = file_path[:-4] + "TRAIN.txt"
    valid_path = file_path[:-4] + "VALID.txt"
    test_path = file_path[:-4] + "TEST.txt"
    with open(train_path, "w") as train:
        with open(valid_path, "w") as valid:
            with open(test_path, "w") as test:
                with open(file_path) as file:
                    for i, line in enumerate(file):
                        if i < ff:
                            if verbose:
                                trmsg = '\rwriting train {} / {}'.format(i,
                                                                         tlen)
                                sys.stdout.write(trmsg)
                                sys.stdout.flush()
                            train.write(line)
                        elif ff <= i < ff + rest:
                            if verbose:
                                vamsg = '\rwriting valid {} / {}'.format(i,
                                                                         tlen)
                                sys.stdout.write(vamsg)
                                sys.stdout.flush()
                            valid.write(line)
                        else:
                            if verbose:
                                temsg = '\rwriting test {} / {}'.format(i,
                                                                        tlen)
                                sys.stdout.write(temsg)
                                sys.stdout.flush()
                            test.write(line)


def clean_and_cut(file_path, verbose=False):
    """
    Function to clean and cut the
    text in train, valid and test.

    :type file_path: str
    :type verbose: boolean
    :rtype: str
    """
    train_path = file_path[:-4] + "CLEANTRAIN.txt"
    valid_path = file_path[:-4] + "CLEANVALID.txt"
    test_path = file_path[:-4] + "CLEANTEST.txt"
    clean_text(file_path)
    file_path = file_path[:-4] + "CLEAN.txt"
    text_cut(file_path, verbose)
    return train_path, valid_path, test_path


def text2folder(file_path, folder_name, verbose=False):
    """
    Given the txt file in "file_path" this function
    separetes four fifths of the lines and write them in the file
    "file_pathTRAIN.txt" (that is why we use the
    variable ff).
    The rest of the text is divide into valid and test txts.

    The 3 files are saved in the folder "folder_name"

    :type file_path: str
    :type folder_name: str
    :type verbose: boolean
    """
    tlen = file_len(file_path)
    ff = int((4 / 5) * tlen)
    rest = tlen - ff
    rest = int(rest / 2)
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)
    train_path = os.path.join(folder_name, "train.txt")
    valid_path = os.path.join(folder_name, "valid.txt")
    test_path = os.path.join(folder_name, "test.txt")
    with open(train_path, "w") as train:
        with open(valid_path, "w") as valid:
            with open(test_path, "w") as test:
                with open(file_path) as file:
                    for i, line in enumerate(file):
                        if i < ff:
                            if verbose:
                                trmsg = '\rwriting train {} / {}'.format(i,
                                                                         tlen)
                                sys.stdout.write(trmsg)
                                sys.stdout.flush()
                            train.write(line)
                        elif ff <= i < ff + rest:
                            if verbose:
                                vamsg = '\rwriting valid {} / {}'.format(i,
                                                                         tlen)
                                sys.stdout.write(vamsg)
                                sys.stdout.flush()
                            valid.write(line)
                        else:
                            if verbose:
                                temsg = '\rwriting test {} / {}'.format(i,
                                                                        tlen)
                                sys.stdout.write(temsg)
                                sys.stdout.flush()
                            test.write(line)


def _read_words(filename):
    """
    Transform a txt file in a list of words. The symbol "\n" is substitute
    by "<eos>" (end of sentence)

    :type filename: str
    :rtype: list
    """
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    """
    Using the _read_words function it reads a txt file
    and creates a dict word: index with size of number
    unique tokens. The most used tokens have lower indexes.

    :type filename: str
    :rtype: dict
    """
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    """
    Transform a txt file into a list of indices using
    word_to_id.

    :type filename: str
    :type word_to_id: dict
    :rtype: list
    """
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def folder2lists(data_path=None):
    """
    Reads text files, converts strings to integer ids
    using the files from data directory "data_path".
    It assumes that this directory contains 3 txt files:
    "train.txt", "valid.txt" and "test.txt".

    :type data_path: str
    :rtype train_data: list
    :rtype valid_data: list
    :rtype test_data: list
    :rtype vocabulary: int
    """
    train_path = os.path.join(data_path, "train.txt")
    valid_path = os.path.join(data_path, "valid.txt")
    test_path = os.path.join(data_path, "test.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


def batch_producer(raw_data, batch_size, num_steps, name=None):
    """
    Iterate on the raw data returning batches of examples x,y.
    Both x and t are shaped [batch_size, num_steps].
    y is the same data time-shifted to the right by one.

    Example:
            raw_data = [4, 3, 2, 1, 0, 5, 6, 1, 1, 1, 1, 0, 3, 4, 1]
            batch_size = 3
            num_steps = 2
            x, y = batch_producer(raw_data, batch_size, num_steps)

            x = [[4, 3], [5, 6], [1, 0]]
            y = [[3, 2], [6, 1], [0, 3]]

    Raises:
        tf.errors.InvalidArgumentError: if batch_size or
        num_steps are too high.

    :type raw_data: list
    :type batch-size: int
    :type num_steps: int
    :type name: str
    :rtype x: tensor
    :rtype y: tensor
    """
    with tf.name_scope(name,
                       "batch_producer",
                       [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data,
                                        name="raw_data",
                                        dtype=tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0: batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        msg = "epoch_size == 0, decrease batch_size or num_steps"
        assertion = tf.assert_positive(epoch_size,
                                       message=msg)
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y
