'''
Created on 2014-12-31

@author: Jonathan
'''
from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
import numpy as np


def main():
    # create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('Data/train.csv', 'r'),
                         delimiter=',',
                         dtype='f8')[1:]
    target = [x[-1] for x in dataset]
    train = [x[:-1] for x in dataset]
    print len([x for x in target if x == 0])
    print len([x for x in target if x == 1])
    test = genfromtxt(open('Data/test.csv', 'r'),
                      delimiter=',',
                      dtype='f8')[1:]
    test_hands = [x[1:] for x in test]
    '''
    create and train the random forest
    multi-core CPUs can use:
        rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    '''
    rf = RandomForestClassifier(n_estimators=10, n_jobs=2)
    rf.fit(train, target)
    result = rf.predict(test_hands)
    to_submit = np.column_stack((range(1, len(result) + 1), result))
    savetxt('Data/submission3.csv', to_submit,
            delimiter=',', fmt='%d')

if __name__ == "__main__":
    main()
