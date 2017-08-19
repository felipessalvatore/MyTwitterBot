# MyTwitterBot: A Twitter bot powered by a [Recurrent Neural Network (RNN)](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) 

[![Build Status](https://travis-ci.org/felipessalvatore/MyTwitterBot.svg?branch=master)](https://travis-ci.org/felipessalvatore/MyTwitterBot)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/felipessalvatore/RrrExample/blob/master/LICENSE)


![alt text](/src/images/robotSmall.png "Robot")

This repository started as an extension of the code of [Assigment 2](http://cs224d.stanford.edu/assignment2/index.html) from [Standford's deep leaning course on NLP](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6). After finishing the assignment I tried to transform the code in something useful (you can judge if I achieve that in any form).

I used a RNN to create a [language model](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf) and with that I created a Twitter bot. There are some corpora in the folder 'data':

- The [Penn Tree Bank (PTB)](https://catalog.ldc.upenn.edu/ldc99t42) dataset.

- All [President Trump](http://www.trumptwitterarchive.com/) sophisticated tweets tweeted so far (02 Aug, 2017). 

- All blog posts from the brazilian jornalist [Leornardo Sakamoto](https://blogdosakamoto.blogosfera.uol.com.br/) posted so far in his site (02 Aug, 2017).

With these corpora I created three different Bots: PTBBot, TrumpBot and SakaBot (not very original names, I know). The general ideia is that you can use all sort of different corpora to generate more creative and fun tweets!

For example, using the PTBBot I tweeted wonderfull things like:

![alt text](/src/images/example.png "example")

## Usage

To install all the required libraries just run:

```
$ sudo apt-get install python3-pip
$ pip3 install -r requirements.txt

```

Before you start creating your own amazing tweets you must first [register your application on Twitter](https://www.youtube.com/watch?v=M7MqML2ZVOY). So, from now on I am assuming that you have all key information in a file called "key.py" (this file should be in the folder "agent").

First, to perform a complete test you can simply run:

```
$ python3 src/test/test_all.py

```

Now, if everything is ok, you can go to the folder 'tutorials' where all bots are located. Let's use the TrumpBot as an example. Before writing any
tweet you need to train the model:

```
$ cd  src/tutorials/TrumpBot
$ python3 train.py

```
After training, you can interact with the bot to write any number of tweets; just run

```
$ python3 write.py

```
All tweets that you wrote from this interaction will be stored on the folder "twitter_draft". Suppose "date.txt" is a text file with some tweets, you can edit this file and then run:

```
$ python3 post.py ./twitter_draft/date.txt -m 30

```
The bot will post all tweets in an interval of 30 minutes using the account that you wrote in the file "key.py". 


## Folders

- **agent**: codes for the bot's behavior.

- **data**: folder with all the corpora.

- **images**: images for the file README.md

- **test**: tests for every module. 

- **text_processing**: different functions for text processing.

- **tftools**: Tensorflow RNN model and helper functions. 

- **tutorials**: Folder with the three basic bots.

- **twitter**: tweepy functions.
