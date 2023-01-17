import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import mean_squared_error


def report(results, n_top=3):
    for i in range(1, n_top+1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print('Model with rank: {0}'.format(i))
            print(
                'Mean validation score: {0:.3f} (std: {1:.3f})'.format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate],
                )
            )
            print('Parameters: {0}'.format(results['params'][candidate]))
            print('')


def retrieve_and_scale_data(indir):
    df = pd.read_csv(indir+os.sep+'train.csv')
    df2 = pd.read_csv(indir+os.sep+'unique_m.csv')
    df2 = df2.drop(df2.columns[-1], axis=1)
    all_data = pd.concat([df, df2], axis=1)
    y = df['critical_temp']
    X = all_data.drop(['critical_temp'], axis=1)
    scale = StandardScaler()
    X_scaled = pd.DataFrame(scale.fit_transform(X), columns=X.columns)
    return X_scaled, y


def main(args):
    X, y = retrieve_and_scale_data(args.indir)
    lassocv = LassoCV(alphas=None, cv=5, max_iter=100000)
    lassocv.fit(X, y)

    model = Lasso()
    model.set_params(alpha=lassocv.alpha_)
    model.fit(X, y)
    print('MSE:', mean_squared_error(y, model.predict(X)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--indir', required=True,
                    help='Input data directory')
    args = ap.parse_args()
    main(args)
