import argparse
import pandas as pd
import numpy as np
#from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def format_data(df):
    labels = ['Source Port', 'NAT Source Port', 'NAT Destination Port']
    df.drop(labels=labels, axis=1, inplace=True)
    df['Destination Port'] = df['Destination Port'].astype('category')
    levels = {'allow': 1, 'drop': 2, 'deny': 3, 'reset-both': 4}
    df['Action'] = df['Action'].replace(levels)
    y = df['Action']
    df.drop(labels=['Action'], axis=1, inplace=True)
    X = df
    #X = StandardScaler().fit_transform(df)
    return X, y


def main(indir):
    df = pd.read_csv(indir)
    #df.drop(labels=['Source Port', 'NAT Source Port'], axis=1, inplace=True)

    #df['Action'] = df['Action'].astype('category')
    #dumb_dest = pd.get_dummies(df['Destination Port'], prefix='DP_')
    #dumb_nat = pd.get_dummies(df['NAT Destination Port'], prefix='NDP_')
    #df.drop(labels=['Destination Port'], axis=1)
    #df = pd.concat([df, dumb_dest, dumb_nat], axis=1)

    #levels = {'allow':0, 'drop':1, 'deny':2, 'reset-both':3}
    #df['Action'] = df['Action'].replace(levels)

    #y = df['Action']
    #df.drop(labels=['Action'], axis=1, inplace=True)
    #X = df
    X, y = format_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    a_vals = np.linspace(15, 17, 2000)

    sgd = SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000, early_stopping=True, random_state=0, shuffle=True)    
    outs = {}
    for al in a_vals:
        try: 
            sgd.alpha = al
            sgd.fit(X_train, y_train)
            preds = sgd.predict(X_test)
            #print('Alpha: ', al)
            #print('Accuracy Score: ', mt.accuracy_score(y_test, preds))
            #print('F1 Score: ', mt.f1_score(y_test, preds, average='weighted'))
            #print('Confusion Matrix: \n', mt.confusion_matrix(y_test, preds))
            outs[ mt.accuracy_score(y_test, preds)] = {'alpha': al , 'accuracy': mt.accuracy_score(y_test, preds), 'F1 Score': mt.f1_score(y_test, preds, average='weighted'), 'Confusion Matrix': mt.confusion_matrix(y_test, preds)}
        except Exception as e:
            pass
    tmp = sorted(outs.keys())
    import pdb
    pdb.set_trace()
    print(tmp[::-1])



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--indir', required=True,
                    help='Please provide input directory with dataset')
    args = ap.parse_args()
    main(args.indir)
