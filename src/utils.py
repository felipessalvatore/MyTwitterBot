import numpy as np
import unittest
import time


def run_test(testClass, header):
    """
    Function to run all the tests from a class of tests.

    :type testClass: unittest.TesCase
    :type header: str
    """
    print(header)
    suite = unittest.TestLoader().loadTestsFromTestCase(testClass)
    unittest.TextTestRunner(verbosity=2).run(suite)


def sample(input_array, temperature=1.0):
    """
    Function to sample an index from a probability array

    :type input_array: np array
    :type header: int
    """
    input_array = np.log(input_array) / temperature
    input_array = np.exp(input_array) / np.sum(np.exp(input_array))
    return np.argmax(np.random.multinomial(1, input_array, 1))


def get_date():
    """
    gives you the date in form:
    day - month - year

    :rtype: str
    """
    return time.strftime('%d-%m-%Y')


def get_real_friends():
    """
    Function that returns a list of all the people on
    Twitter that I really want to follow. You can follow them two !!!

    :rtype: str
    """
    RealFriends = [2759214930, 737739260044840961, 527038338, 15825547,
                   842892190024130560, 816825631, 869813078, 15595161,
                   156557697, 429298107, 145344850, 725383130517897216,
                   39547749, 8493302, 154101116, 775449094739197953,
                   1132112042, 762434885902491649, 63802808, 14903327,
                   20609518, 798297731961913344, 318879680, 2877269376,
                   2895770934, 9088102, 440109717, 564919357, 40838623,
                   19879313, 241382835, 237870401, 208711264, 18893015,
                   783356145670852608, 426267558, 18379614, 3270678680,
                   52728478, 15278016, 5854882, 37710752, 19725644,
                   181228051, 254107028, 3092521983, 3973622834,
                   815095933, 116994659, 2785337469, 2304815288, 3120987492,
                   5841, 611941828, 478782624, 2898824212, 16367706, 304679484,
                   299206560, 358881865, 167346791, 634406525, 15804774,
                   34743251, 13298072, 30817647, 120288032, 318063815,
                   1491097802, 2865941804, 11518572, 361334185, 33838201,
                   48008938, 4783690002, 1482581556, 130745589, 216939636,
                   18825961, 17174309, 15661871, 15962096, 1881225786,
                   453574328, 4194417913, 27022632, 28406270, 20035593,
                   2821417461, 197313522, 118263124, 29843511, 2815077014,
                   50090898, 13334762, 3333052551, 1442906958, 111312320,
                   15977916, 99661289, 117539435, 4620451, 386693228,
                   4398626122, 46453032, 18638502, 44196397, 2999992556,
                   15808647, 1344951, 15271933, 3015811923, 14768363,
                   5633002, 215696856, 715015710, 34546946, 19566708,
                   20322986, 682463, 321498783, 546197366, 37876253,
                   175624200, 14114410, 82450716, 38265197, 316572265,
                   38297956, 37965564, 42754483, 119958446, 107336879,
                   14587578, 79522134, 15439395, 24585498, 8547012]

    return RealFriends
