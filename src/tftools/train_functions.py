import time
import sys
import tensorflow as tf
import numpy as np
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from text_processing.functions import ptb_iterator


def run_epoch(model,
              session,
              data,
              train_op=None,
              verbose=10):
    """
    Use the tf graph of the model to run one epoch
    through the data using the tf session.
    Note that we need to pass in the initial state and
    retrieve the final state to give the RNN proper history

    :type model: RNNLanguageModel
    :type session: tf Session
    :type data: np ndarray
    :type train_op: None or tf Tensor
    :type verbose: int
    :rtype: float
    """
    config = model.config
    dp = config.dropout
    if not train_op:
        train_op = tf.no_op()
        dp = 1
    DataIte1 = ptb_iterator(data, config.batch_size, config.num_steps)
    DataIte2 = ptb_iterator(data, config.batch_size, config.num_steps)
    total_steps = sum(1 for _ in DataIte1)
    total_loss = []
    state = session.run(model.initial_state)
    for step, (x, y) in enumerate(DataIte2):
        feed = {model.input_placeholder: x,
                model.labels_placeholder: y,
                model.initial_state: state,
                model.dropout_placeholder: dp}
        loss, state, _ = session.run([model.loss, model.final_state, train_op],
                                     feed_dict=feed)
        total_loss.append(loss)
        if verbose and step % verbose == 0:
            sys.stdout.write('\r{} / {} : pp = {}'.format(step,
                                                          total_steps,
                                                          np.exp(np.mean(total_loss))))
            sys.stdout.flush()
    if verbose:
        sys.stdout.write('\r')
    return np.exp(np.mean(total_loss))


def train_model(model,
                save=True,
                debug=False):
    """
    Use the tf graph of the model to train the model

    :type model: RNNLanguageModel
    :type save: boolean
    :type debug: boolean
    """
    config = model.config
    if debug:
        max_epochs = 1
    else:
        max_epochs = config.max_epochs
    best_val_pp = float('inf')
    best_val_epoch = 0
    with tf.Session(graph=model.graph) as sess:
        tf.global_variables_initializer().run()
        for epoch in range(max_epochs):
            print(('Epoch {}'.format(epoch)))
            start = time.time()
            train_pp = run_epoch(model,
                                 sess,
                                 model.encoded_train,
                                 model.train_op)
            valid_pp = run_epoch(model,
                                 sess,
                                 model.encoded_valid)
            print(('Training perplexity: {}'.format(train_pp)))
            print(('Validation perplexity: {}'.format(valid_pp)))
            if valid_pp < best_val_pp:
                    best_val_pp = valid_pp
                    best_val_epoch = epoch
                    if save:
                        model.saver.save(sess, model.save_path)
            if epoch - best_val_epoch > config.early_stopping:
                break
            print(('Total time: {}'.format(time.time() - start)))
