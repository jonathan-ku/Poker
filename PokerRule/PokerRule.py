'''
Created on 2014-12-31

@author: Jonathan
'''
from sklearn.ensemble import RandomForestClassifier
from itertools import permutations
import numpy as np


def get_train():
    # create the training sets, skipping the header row with [1:]
    print "Reading training data..."
    dataset = np.genfromtxt(open('Data/train.csv', 'r'),
                            delimiter=',',
                            dtype='f8')[1:]
    return dataset


def get_test():
    # create the test sets, skipping the header row with [1:]
    print "Reading test data..."
    test = np.genfromtxt(open('Data/test.csv', 'r'),
                         delimiter=',',
                         dtype='f8')[1:]
    return [x[1:] for x in test]


def encode_data_suit(dataset):
    '''
    Encode the suit
    '''
    print "Encoding data..."
    to_return = []
    for hand in dataset:
        temp_hand = []
        for i in xrange(0, len(hand)-1):
            if i % 2 == 0:
                suit = hand[i]
                if suit == 1:
                    temp_hand.append(14)
                elif suit == 2:
                    temp_hand.append(15)
                elif suit == 3:
                    temp_hand.append(16)
                else:
                    temp_hand.append(17)
            else:
                temp_hand.append(hand[i])
        temp_hand.append(hand[-1])
        to_return.append(temp_hand)
    return to_return


def encode_data_card_num(dataset):
    '''
    Encode into card number
    '''
    to_return = []
    for hand in dataset:
        temp_hand = []
        for i in xrange(0, len(hand)-1, 2):
            temp_hand.append(hand[i]*hand[i+1])
        temp_hand.append(hand[-1])
        to_return.append(temp_hand)
    return to_return


def generate_permutation(dataset):
    print "Generating permutation..."
    to_return = []
    for hand in dataset:
        for order in permutations("02468"):
            temp_hand = []
            for i in order:
                index = int(i)
                temp_hand.append(hand[index])
                temp_hand.append(hand[index+1])
            temp_hand.append(hand[-1])
            to_return.append(temp_hand)
    return to_return


def random_forest(train, target, test, test_ans, n):
    '''
    create and train the random forest
    '''
    rf = RandomForestClassifier(n_estimators=n, n_jobs=2)
    rf.fit(train, target)
    print "%d    %f" % (n, rf.score(test, test_ans))


def generate_submit(result):
    print "Generating submission file..."
    to_submit = np.column_stack((range(1, len(result) + 1), result))
    np.savetxt('Data/submission4.csv', to_submit,
               delimiter=',', fmt='%d')


def main():
    raw_dataset = get_train()
    dataset = encode_data_suit(raw_dataset)
    dataset = generate_permutation(dataset)
    # print dataset[0:10]
    # '''
    target = [x[-1] for x in dataset]
    train = [x[:-1] for x in dataset]
    test = get_test()
    test_encoded = encode_data_suit(test)
    print "Init Random Forest..."
    rf = RandomForestClassifier(n_estimators=110, n_jobs=2)
    print "Fitting Random Forest..."
    rf.fit(train, target)
    print "Done fitting..."
    generate_submit(rf.predict(test_encoded))
    '''
    target = [x[-1] for x in dataset[:-2000]]
    train = [x[:-1] for x in dataset[:-2000]]
    test_ans = [x[-1] for x in dataset[-2000:]]
    test_hands = [x[:-1] for x in dataset[-2000:]]
    for n in xrange(10, 310, 10):
        random_forest(train, target, test_hands, test_ans, n)
    '''

if __name__ == "__main__":
    main()
