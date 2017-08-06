from Config import Config
from DataHolder import DataHolder
from RNNLanguageModel import RNNLanguageModel
from train_functions import train_model
from generate_functions import generate_loop
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)


def main(text_path, train=False):
    """
    script to enter the generator generator loop

    :type train: boolean
    :type text_path: str
    """
    config = Config(max_epochs=2)
    if train:
        data = DataHolder(text_path=text_path)
        model = RNNLanguageModel(config, data)
        train_model(model)
    generate_loop(config, ShowTest=False)


if __name__ == "__main__":
    text_path = os.path.join(parentdir, "data")
    DataHolder(text_path=text_path)
    main(text_path=text_path, train=True)
