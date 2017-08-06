from collections import defaultdict


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
