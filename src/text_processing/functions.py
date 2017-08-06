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
    with the string  "", every number with "N", every emoji with "".Multiple
    spaces are also eliminated. It also puts evey word in the lower case format

    :type path: str
    """
    new_path = path[:-4] + "CLEAN.txt"
    url = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    nums = re.compile(r'[+-]?\d+(?:\.\d+)?')
    punct = re.compile(r'[.?\-",!;–…]+')
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
            new_line = url.sub("", line)
            new_line = nums.sub("N", new_line)
            new_line = punct.sub(" ", new_line)
            new_line = emoji_pattern.sub(" ", new_line)
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
    text_size = file_len(file_path)
    ff = int((4 / 5) * text_size)
    rest = text_size - ff
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
                                sys.stdout.write('\rwriting train {} / {}'.format(i,text_size))
                                sys.stdout.flush()
                            train.write(line)
                        elif ff <= i < ff + rest:
                            if verbose:
                                sys.stdout.write('\rwriting valid {} / {}'.format(i,text_size))
                                sys.stdout.flush()
                            valid.write(line)
                        else:
                            if verbose:
                                sys.stdout.write('\rwriting test {} / {}'.format(i,text_size))
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
