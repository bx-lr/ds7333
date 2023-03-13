import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt
from sklearn import svm


def main(indir):
    df = pd.read_csv(indir)
    df.drop(labels=['Source Port', 'NAT Source Port', 'NAT Destination Port'], axis=1, inplace=True)

    #df['Action'] = df['Action'].astype('category')
    #dumb_dest = pd.get_dummies(df['Destination Port'], prefix='DP_')
    #dumb_nat = pd.get_dummies(df['NAT Destination Port'], prefix='NDP_')
    #df = pd.concat([df, dumb_dest, dumb_nat], axis=1)
    #df.drop(labels=['Destination Port'], axis=1)
    #df = pd.concat([df, dumb_dest], axis=1)
    df['Destination Port'] = df['Destination Port'].astype('category')

    levels = {'allow':1, 'drop':2, 'deny':3, 'reset-both':4}
    df['Action'] = df['Action'].replace(levels)
    y = df['Action']
    df.drop(labels=['Action'], axis=1, inplace=True)
    X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60,
                                                        random_state=0)
    c_vals = np.logspace(-20, 20, 20)
    print('Tuning C...')
    for C in c_vals:
        svm_clf = svm.SVC(C=C, kernel='linear',  gamma='scale')
        svm_clf.fit(X_train, y_train)
        y_hat = svm_clf.predict(X_test)
        accuracy = mt.accuracy_score(y_test, y_hat)
        f1 = mt.f1_score(y_test, y_hat, average='weighted')
        conf_mat = mt.confusion_matrix(y_test, y_hat)
        print("accuracy for C={}: {}".format(C, accuracy))
        print("F1 Score for C={}: {}".format(C, f1))
        print("confusion matrix\n cor C={}:\n {}\n".format(C, conf_mat))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--indir', required=True,
                    help='Please provide input directory with dataset')
    args = ap.parse_args()
    main(args.indir)
