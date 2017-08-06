class Config(object):
    """
    Holds model hyperparams and data information.

    :type embed_size: int
    :type batch_size: int
    :type num_steps: int
    :type hidden_size: int
    :type max_epochs: int
    :type early_stopping: int
    :type dropout: float
    :type lr: float
    """
    def __init__(self,
                 embed_size=50,
                 batch_size=104,
                 num_steps=14,
                 hidden_size=100,
                 max_epochs=16,
                 early_stopping=2,
                 dropout=0.991323729933,
                 lr=0.00217346380124):
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.dropout = dropout
        self.lr = lr
