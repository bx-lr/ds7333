import argparse
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt
from sklearn import svm
from sklearn.linear_model import SGDClassifier


def do_svm(X, y, C=0.01):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60,
                                                        random_state=0)
    svm_clf = svm.SVC(C=C, kernel='linear',  gamma='scale')
    svm_clf.fit(X_train, y_train)
    y_hat = svm_clf.predict(X_test)
    accuracy = mt.accuracy_score(y_test, y_hat)
    f1 = mt.f1_score(y_test, y_hat, average='weighted')
    conf_mat = mt.confusion_matrix(y_test, y_hat)
    print('SVM Results:')
    print("Accuracy for C={}: {}".format(C, accuracy))
    print("F1 Score for C={}: {}".format(C, f1))
    print("Confusion matrix\n cor C={}:\n {}\n".format(C, conf_mat))
    print('')


def do_sgd(X, y, al=0.0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                        random_state=0)
    sgd = SGDClassifier(loss='log_loss',
                        penalty='l2',
                        alpha=al,
                        max_iter=1000,
                        early_stopping=True,
                        random_state=0,
                        shuffle=True)
    sgd.fit(X_train, y_train)
    preds = sgd.predict(X_test)
    print('SGD Results:')
    print('Accuracy:', mt.accuracy_score(y_test, preds))
    print('F1 Score:', mt.f1_score(y_test, preds, average='weighted'))
    print('Confusion Matrix:\n', mt.confusion_matrix(y_test, preds))
    print('')


def format_data(df):
    labels = ['Source Port', 'NAT Source Port', 'NAT Destination Port']
    df.drop(labels=labels, axis=1, inplace=True)
    df['Destination Port'] = df['Destination Port'].astype('category')
    levels = {'allow': 1, 'drop': 2, 'deny': 3, 'reset-both': 4}
    df['Action'] = df['Action'].replace(levels)
    y = df['Action']
    df.drop(labels=['Action'], axis=1, inplace=True)
    X = df
    return X, y


def main(infile):
    if not os.path.exists(os.path.abspath(infile)):
        print('Cant find input file... exiting')
        sys.exit(-1)
    df = pd.read_csv(infile)
    X, y = format_data(df)
    print('Training SVM...')
    do_svm(X, y, C=0.000695)
    print('Training SGD...')
    do_sgd(X, y, al=15.183091545772886)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--infile', required=True,
                    help='Please provide input file')
    args = ap.parse_args()
    main(args.infile)
