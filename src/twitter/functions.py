def TweetValid(tweet, CharNumber=140):
    """
    Function to check if a string "tweet" is valid,
    i.e., the #char is equal or less than CharNumber.
    CharNumber can be 140 at maximum.

    :type tweet: str
    :type CharNumber: int
    :rtype: boolean
    """
    if CharNumber > 140:
        CharNumber = 140
    return len(tweet) <= CharNumber


def eos2period(word):
    """
    Changing all the '<eos>' for '.'

    :type word: str
    :rtype: str
    """
    if word == '<eos>':
        return '.'
    else:
        return word
