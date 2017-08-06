# MyTwitterBot: A Twitter bot powered by a [Recurrent Neural Network (RNN)](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) 

[![Build Status](https://travis-ci.org/felipessalvatore/MyTwitterBot.svg?branch=master)](https://travis-ci.org/felipessalvatore/MyTwitterBot)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/felipessalvatore/RrrExample/blob/master/LICENSE)


![alt text](/src/images/robotSmall.png "Robot")

This repository started as an extension of the code of [Assigment 2](http://cs224d.stanford.edu/assignment2/index.html) from [Standford's deep leaning course on NLP](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6).

I used a RNN to create a [language model](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf). The Twitter bot is based on this language model. There is some corpora in the folder data. But the default one is the [Penn Tree Bank (PTB)](https://catalog.ldc.upenn.edu/ldc99t42) dataset. You can use different corpora to generate more creative and fun tweets.

Using the PTB corpus I can create wonderfull tweets like this one:

![alt text](/src/images/example.png "example")

## Usage

To install all the required libraries just run:

```
$ sudo apt-get install python3-pip
$ pip3 install -r requirements.txt

```

Before you start creating your own amazing tweets you must first [register your application on Twitter](https://www.youtube.com/watch?v=M7MqML2ZVOY). I am assuming that you have all key information in a file called "key.py"
(this file should be in the folder "agent").

To perform a complet test you can simply run:

```
$ python3 src/test/test_all.py

```

## Folders

- **agent**: codes for the bot's behavior.

- **data**: folder with the PTB corpus devided in train, test and valid.

- **images**: images for the file README.md

- **test**: tests for every module. 

- **text_processing**: different functions for text processing.

- **tftools**: Tensorflow RNN model and helper functions. 

- **tutorials**: Jupyter notebooks to present the Bot.

- **twitter**: tweepy functions.