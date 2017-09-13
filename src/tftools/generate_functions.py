from utils import sample
from copy import deepcopy
import tensorflow as tf

try:
    from train_functions import run_epoch
    from RNNLanguageModel import RNNLanguageModel
    from DataHolder import DataHolder
except ImportError:
    from tftools.train_functions import run_epoch
    from tftools.RNNLanguageModel import RNNLanguageModel
    from tftools.LSTMLanguageModel import LSTMLanguageModel
    from tftools.DataHolder import DataHolder


def generate_text(session, model, config, starting_text='<eos>',
                  stop_length=100, stop_tokens=None, temp=1.0, lstm=False):
    """
    This function uses the model to generate a sentence
    starting with the token(s) "starting_text".
    The generated sentence has at most "stop_length" tokens.
    If you use the list "stop_tokens", the sentence will end at any
    word of that list.

    :type session: tf Session
    :type model: RNNLanguageModel
    :type config: Config
    :type starting_text: str
    :type stop_lenght: int
    :type stop_tokens: None or list of str
    :type temp: float
    :rtype : list of str
    """
    if not lstm:
        state = session.run(model.initial_state)
    tokens = [model.vocab.encode(word) for word in starting_text.split()]
    for i in range(stop_length):
        if not lstm:
            feed = {model.input_placeholder: [[tokens[-1]]],
                    model.initial_state: state,
                    model.dropout_placeholder: 1.0}
        else:
            feed = {model.input_placeholder: [[tokens[-1]]],
                    model.dropout_placeholder: 1.0}
        state, y_pred = session.run([model.final_state,
                                     model.predictions[-1]],
                                    feed_dict=feed)
        next_word_idx = sample(y_pred[0], temperature=temp)
        tokens.append(next_word_idx)
        if stop_tokens and model.vocab.decode(tokens[-1]) in stop_tokens:
            break
    output = [model.vocab.decode(word_idx) for word_idx in tokens]
    return output


def generate_sentence(session, model, config, *args, **kwargs):
    """
    Convenience function. Similar to generate_text

    :type session: tf Session
    :type model: RNNLanguageModel
    :type config: Config
    :rtype : list of str
    """
    return generate_text(session,
                         model,
                         config,
                         *args,
                         stop_tokens=['<eos>'],
                         **kwargs)


def generate_loop(config, text_path, ShowTest=True, lstm=False):
    """
    Genereate sentences in the command line
    until the user type "*end*"

    :type config: Config()
    :type text_path: str
    :type ShowTest: boolean
    """
    gen_config = deepcopy(config)
    gen_config.batch_size = gen_config.num_steps = 1
    dataholder = DataHolder(text_path=text_path)
    if not lstm:
        model = RNNLanguageModel(gen_config, dataholder)
    else:
        model = LSTMLanguageModel(gen_config, dataholder)
    with tf.Session(graph=model.graph) as sess:
        model.saver.restore(sess, model.save_path)
        if ShowTest:
            test_pp = run_epoch(model, sess, model.encoded_test, lstm=lstm)
            print(('=-=' * 5))
            print(('Test perplexity: {}'.format(test_pp)))
            print(('=-=' * 5))
            print(' ')
        print(('=-=' * 5))
        print("Sentence generator\nType '*end*' to break the loop")
        print(('=-=' * 5))
        starting_text = 'i am'
        while starting_text != "*end*":
            print(' '.join(generate_sentence(sess,
                                             model,
                                             config,
                                             starting_text=starting_text,
                                             temp=1.0)))
            starting_text = input('> ')
