try:
    from functions import TweetValid, eos2period
except ImportError:
    from twitter.functions import TweetValid, eos2period
from copy import deepcopy, copy
import tensorflow as tf
import os
import numpy as np
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from tftools.Config import Config
from tftools.DataHolder import DataHolder
from tftools.RNNLanguageModel import RNNLanguageModel
from tftools.train_functions import train_model
from utils import sample


class TweetGenerator():
    """
    Class that generates tweets. You need to train the
    model before generate any tweet. "black_list" is
    the list of all words that the bot should not say.
    The list of noums is always non empty. In the worst
    case scenario (when balck_list = all_noums) we set
    all_noums to "[empty list]"

    :type text_path: srt
    :type config: Config
    :type train: boolean
    :type debug: boolean
    :type black_list: list
    """
    def __init__(self,
                 text_path,
                 config=None,
                 train=False,
                 debug=False,
                 black_list=[]):
        self.black_list = black_list
        self.is_trained = not train
        self.dataholder = DataHolder(text_path=text_path, debug=debug)
        self.dataholder.all_noums = [word for word in self.dataholder.all_noums
                                     if word not in black_list]
        if self.dataholder.all_noums == []:
            self.dataholder.all_noums.append("empty list")
        if config is None:
            self.config = Config()
        else:
            self.config = config
        if not train:
            save_path = os.path.join(os.getcwd(), "checkpoints")
            if not os.path.exists(save_path):
                self.train()
        else:
            self.train()

    def train(self):
        """
        Training the model
        """
        print("==Training the model==")
        model = RNNLanguageModel(self.config,
                                 self.dataholder)
        train_model(model)
        self.is_trained = True

    def __generate_tweet_no_unk__(self,
                                  session,
                                  model,
                                  config,
                                  starting_text='<eos>',
                                  stop_tokens=None,
                                  temp=1.0,
                                  CharSize=140):
        """
        Private method to generate a sentence.
        The sentence will have at maximun 140 characters (a tweet).
        We use the list of all noums from
        the vocav to eliminate all unk tokens that may occur.

        :type session: tf Session
        :type model: RNNLanguageModel
        :type config: Config
        :type starting_text: str
        :type stop_tokens: None or list of str
        :type temp: float
        :rtype : list of str
        """
        vocab = self.dataholder.vocab
        state = session.run(model.initial_state)
        tweet = starting_text.split()
        tweet_as_str = starting_text
        tokens = [vocab.encode(word) for word in starting_text.split()]
        while True:
            feed = {model.input_placeholder: [[tokens[-1]]],
                    model.initial_state: state,
                    model.dropout_placeholder: 1.0}
            state, y_pred = session.run([model.final_state,
                                         model.predictions[-1]],
                                        feed_dict=feed)
            next_word_idx = sample(y_pred[0], temperature=temp)
            condit1 = vocab.decode(next_word_idx) == self.dataholder.unk_token
            condit2 = vocab.decode(next_word_idx) in self.black_list
            if condit1 or condit2:
                choice = np.random.choice(len(self.dataholder.all_noums), 1)[0]
                next_word = self.dataholder.all_noums[choice]
            else:
                next_word = vocab.decode(next_word_idx)
            before_next_word = copy(tweet)
            tokens.append(next_word_idx)
            tweet.append(next_word)
            tweet_as_str = " ".join(tweet)
            if len(tweet_as_str) == CharSize:
                break
            if not TweetValid(tweet_as_str, CharNumber=CharSize):
                tweet = copy(before_next_word)
                break
            if stop_tokens and vocab.decode(tokens[-1]) in stop_tokens:
                break
        return tweet

    def generate_tweet_list(self,
                            number_of_tweets=1,
                            starting_text='<eos>',
                            hashtag_list=[]):
        """
        Given the words in the string "starting text"
        this method generates "number_of_tweets" tweets.
        It also appends at the end of the sentence
        the hashtags that may occur in the list "hashtag_list".

        :type number_of_tweets: int
        :type starting_text: str
        :type hashtag_list: list of str
        :rtype: list of str
        """
        cdr = " ".join(hashtag_list)
        cdr_size = len(cdr) + 1
        if starting_text == '<eos>':
            text_so_far = cdr
        else:
            text_so_far = starting_text + " " + cdr
        assert TweetValid(text_so_far,
                          CharNumber=140), "Equal or less than 140 characters!"
        size = 140 - cdr_size
        all_tweets = []
        gen_config = deepcopy(self.config)
        gen_config.batch_size = gen_config.num_steps = 1
        gen_model = RNNLanguageModel(gen_config,
                                     self.dataholder)
        with tf.Session(graph=gen_model.graph) as sess:
                gen_model.saver.restore(sess, gen_model.save_path)
                for i in range(number_of_tweets):
                    tweet = self.__generate_tweet_no_unk__(sess,
                                                           gen_model,
                                                           gen_config,
                                                           starting_text,
                                                           CharSize=size)
                    tweet = " ".join([eos2period(word) for word in tweet])
                    if hashtag_list != []:
                        tweet = tweet + " " + cdr
                    all_tweets.append(tweet)

        return all_tweets
