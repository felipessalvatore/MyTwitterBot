import os
import tensorflow as tf

try:
    from basic_functions import init_wb, affine_transformation
except ImportError:
    from tftools.basic_functions import init_wb, affine_transformation


class LSTMLanguageModel():
    """
    Language model based on a LSTM RNN

    :type config: Config
    :type dataholder: DataHolder
    :type debug: boolean
    :type search: boolean
    """

    def __init__(self, config, dataholder, debug=False, search=False):
        self.config = config
        self.num_steps = self.config.num_steps
        self.embed_size = self.config.embed_size
        self.batch_size = self.config.batch_size
        self.hidden_size = self.config.hidden_size
        self.search = search
        self.vocab = dataholder.vocab
        self.vocab_size = len(self.vocab)
        self.encoded_train = dataholder.encoded_train
        self.encoded_valid = dataholder.encoded_valid
        self.encoded_test = dataholder.encoded_test
        self.build_graph()

    def add_placeholders(self):
        """
        Adding placeholders for the graph

        """
        input_shape = [self.batch_size, self.num_steps]
        self.input_placeholder = tf.placeholder(tf.int32,
                                                shape=input_shape,
                                                name="input_placeholder")
        self.labels_placeholder = tf.placeholder(tf.int32,
                                                 shape=input_shape,
                                                 name="labels_placeholder")
        self.dropout_placeholder = tf.placeholder(tf.float32,
                                                  shape=[],
                                                  name="dropout_value")

    def add_embedding(self):
        """
        Add embedding layer.

        L is the matrix of all word embeddings --
        L.shape = (self.vocab_size, embed_size)

        Remember, in this case a batch is just a collection
        of sub-lines of the corpus (this collection has size
        "batch_size"). Each sub-line is an array of ints of
        size "num_steps" (each int represents a word in the vocabulary).
        When we apply tf.nn.embedding_lookup(L, input_placeholders)
        we select the word vector of each word in each sub-line of the batch.
        The result of this selection is the tensor "look" --
        look.shape = ("batch_size", "num_steps", "embed_size").
        In an informal way, we can write

        look = [sub-line of word vectors[0], ...,
                sub-line of word vectors[batch_size -1]]

        We use the functions tf.split and tf.squeeze to change
        the tensor look and create a list "input". Such that:
            - len(input) = num_steps
            - tensor.shape = (batch_size, embed_size), for tensor in input

        We can think on this list
        as follows: input[i] is the tensor that have all the word embeddings
        in the entry i from each sub-line in look.
        """
        with tf.variable_scope("WordEmbeddings"):
            Lshape = (self.vocab_size, self.embed_size)
            self.L = tf.get_variable("L", shape=Lshape)
            self.look = tf.nn.embedding_lookup(self.L, self.input_placeholder)
            self.split = tf.split(self.look, self.num_steps, 1)
            self.inputs = [tf.squeeze(tensor, squeeze_dims=[1])
                           for tensor in self.split]

    def add_logits(self):
        """
        The "recurrent" part of the model is the following:

        We use the tensor "initial_state" to
        start the loop. For each tensor t in the list "inputs" we:

            - calculate the tensor
              h = sigmoid(previous_h*H + (t*weights + bias)).

            - store the tensor h in the list rnn_outputs.

            - set previous_h to be h.

            - and in the last iteration we store h as the tensor "final_state".

        At each iteration i, h_i is a matrix where h_i[j] is the memory
        of the word i in the batch j.

        The result of the loop is the list "rnn_outputs" such that
            - len(rnn_outputs) = num_steps
            - tensor.shape = (batch_size, embed_size),for tensor in rnn_outputs

        At the output layer we will apply an affine transformation on
        each tensor of the list rnn_outputs. The list of all these tensors
        is called "logits":
            - len(logits) = num_steps
            - tensor.shape = (batch_size, self.hidden_size) for tensor in logits

        """
        # initialshape = (self.batch_size, self.hidden_size)
        # self.initial_state = tf.zeros(initialshape)
        lstmCell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
        # print("AAAAAAAAAAAAAAAAAa",lstmCell.state_size)
        # self.rnn_outputs, self.final_state = lstmCell.__call__(self.inputs,
        #                                                        self.initial_state)
        self.rnn_outputs, self.final_state = tf.nn.static_rnn(lstmCell,
                                                              self.inputs,
                                                              dtype=tf.float32)

        Vshape = (self.config.hidden_size, self.vocab_size)

        with tf.variable_scope("Projection_layer"):
            self.output_weights = init_wb(Vshape, "output_weights")
            self.logits = [affine_transformation(tensor, self.output_weights)
                           for tensor in self.rnn_outputs]

    def add_prediction(self):
        """
        This method transforms every tensor from "logits",
        in a distribution over the vocabulary using the softmax function.
        The list of distribution is called "predictions" (same shape as
        the list "logits").
        """
        self.predictions = [tf.nn.softmax(tf.cast(tensor, 'float64'))
                            for tensor in self.logits]

    def add_loss(self):
        """
        The tensor "logitsReshaped1" concatenate every tensor from the
        list "logits". So:

        logitsReshaped1[i] = tensor[0] (con) ... (con) tensor[num_steps]

            - (con) is the concatenation operation
            - where tensor[k] is logits[k][i]

        And "logitsReshaped2" will slice all these concatenated tensors in one
        tensor of shape (num_steps * batch_size, vocab_size).

        A exemple is usefull to undestaing all the shape transformations.

        Suppose num_steps = 2 and batch_size = 3. Then, self.input_placeholder
        will be of the form:

        self.input_placeholder = [[w11 w12],
                                  [w21 w22],
                                  [w31 w32]]

            - self.input_placeholder.shape = (batch_size, num_steps)

        Let  wij* be the wordvectors for wij. "self.inputs" is of the form

        self.inputs = [[w11*, w21*, w31*],
                       [w12*, w22*, w32*]]

            - self.inputs.shape = (num_steps, batch_size, embed_size)

        Let logit(wij*) be the result of applying the model to wij*.
        Then "self.logits" is of the form:

        self.logits = [[logit(w11*), logit(w21*), logit(w31*)],
                       [logit(w12*), logit(w22*), logit(w32*)]]

            - self.logits.shape = (num_steps, batch_size, vocab_size)

        Let (con) be the concatenation operation. Then,

        self.logitsReshaped1 = [logit(w11*)(con)logit(w12*),
                               logit(w21*)(con)logit(w22*),
                               logit(w31*)(con)logit(w32*)]

            - self.logitsReshaped1.shape = (batch_size, num_steps * vocab_size)

        And finally self.logitsReshaped2 is:

        self.logitsReshaped2 = [logit(w11*),
                                logit(w12*),
                                logit(w21*),
                                logit(w22*),
                                logit(w31*),
                                logit(w32*)]

            - self.logitsReshaped2.shape = (num_steps * batch_size, vocab_size)

        We have a similar case for the tensor self.labelsReshaped. Using
        the params of the example above, we have:

         self.labels_placeholder =[[v11 v12],
                                  [v21 v22],
                                  [v31 v32]]

            - self.labels_placeholder.shape = (batch_size, num_steps)

        After the reshape:

        self.labelsReshaped = [v11,
                               v12,
                               v21,
                               v22,
                               v31,
                               v32]

            - self.labelsReshaped.shape = (num_steps * batch_size)

        """
        self.logitsReshaped1 = tf.concat(self.logits, 1)
        self.logitsReshaped2 = tf.reshape(self.logitsReshaped1,
                                          [-1, self.vocab_size])
        self.labelsReshaped = tf.reshape(self.labels_placeholder,
                                         [self.batch_size *
                                          self.num_steps])
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labelsReshaped,
                                                                   logits=self.logitsReshaped2)
        self.loss = tf.reduce_mean(self.loss)

    def add_training_op(self):
        """
        Method to create the graph optimizer.
        """
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        self.train_op = optimizer.minimize(self.loss)

    def add_saver(self):
        """
        Method to create the graph saver.
        """
        self.saver = tf.train.Saver()
        save_dir = 'checkpoints/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_path = os.path.join(save_dir, 'best_validation')

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.add_placeholders()
            self.add_embedding()
            self.add_logits()
            self.add_prediction()
            self.add_loss()
            self.add_training_op()
            self.add_saver()
