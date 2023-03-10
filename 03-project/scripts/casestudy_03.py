import argparse
import glob
import os
import email
import traceback
import queue
import warnings
import pandas as pd
import numpy as np
import copy as cp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score


def get_payload_data(payloads, em):
    out = []
    work = queue.Queue()
    _ = [work.put(p) for p in payloads]
    while not work.empty():
        p = work.get()
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, list):
            for item in p:
                if type(item) == type(em):
                    work.put(item)
                if isinstance(item, str):
                    out.append(item)
        else:
            tmp = p.get_payload()
            for t in tmp:
                if type(t) == type(em):
                    work.put(t)
                if isinstance(t, str):
                    out.append(t)
    return out


def collect_data_from_messages(indir):
    files = [os.path.abspath(f)
             for f in glob.glob(indir+os.path.sep+'**', recursive=True)
             if os.path.isfile(f)]
    output = []
    for fname in files:
        f_dict = {}
        with open(fname, 'r', encoding='latin-1') as fd:
            try:
                f_dict['fpath'] = fname
                em = email.message_from_string(fd.read())
                f_dict['subject'] = em['Subject']
                f_dict['payload'] = em.get_payload()
                f_dict['contenttype'] = em.get_content_type()

                if f_dict['contenttype'].find('multipart') > -1:
                    payloads = f_dict['payload']
                    if isinstance(payloads, list):
                        tmp = [p.get_payload() for p in payloads]
                        tmp = get_payload_data(tmp, em)
                        f_dict['payload'] = ''.join(tmp)
                    elif type(payloads) == type(em):
                        tmp = payloads.get_payload()
                        f_dict['payload'] = tmp
                    else:
                        pass
                msg_class = fname.split(os.sep)[-2]
                f_dict['isSpam'] = 1 if msg_class.find('ham') < 0 else 0
            except Exception as e:
                print('error processing file:', fname)
                print(traceback.format_exc())
                print(e)
        output.append(f_dict)
    df = pd.DataFrame(output)
    return df


def remove_special(df):
    remove = ['\n', '\t', '\r', '\\n', '\\t', '\\r', '!', '@', '#', '$',
              '%', '^', '&', '*', '(', ')', '-', '_', '+', '+', '{', '}',
              '[', ']', '\\', '|', ':', ';', '"', "'", ',', '.', '/', '?',
              '<', '>', '`', '~']
    tmp = df['payload']
    for r in remove:
        tmp = tmp.str.replace(r, '')
    df['payload'] = tmp
    return df


def vectorize_features(df):
    vect = CountVectorizer(lowercase=True, stop_words='english')
    cv_data = vect.fit_transform(df['payload'])
    count_vect_df = pd.DataFrame(cv_data.todense(),
                                 columns=vect.get_feature_names_out())
    combined_df = pd.concat([df, count_vect_df], axis=1)
    combined_df.drop(['fpath', 'subject'], axis=1, inplace=True)
    return combined_df


def get_kmeans_labels(combined_df):
    sil = []
    tmp = combined_df.drop(['isSpam', 'payload', 'contenttype'], axis=1)
    kmeans = KMeans(n_clusters=2).fit(tmp)
    labels = kmeans.labels_
    sil.append(silhouette_score(tmp, labels, metric='euclidean'))
    print('\tSilhouette score: ', sil)
    label_df = pd.DataFrame({'labels': labels})
    final_df = pd.concat([combined_df, label_df], axis=1)
    return final_df


def cross_val_predict(model, kfold, X, y):
    model_ = cp.deepcopy(model)
    no_classes = len(np.unique(y))
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes])
    for train_ndx, test_ndx in kfold.split(X):
        train_X = X[train_ndx]
        train_y = y[train_ndx]
        test_X = X[test_ndx]
        test_y = y[test_ndx]
        actual_classes = np.append(actual_classes, test_y)
        model_.fit(train_X, train_y)
        predicted_classes = np.append(
                                      predicted_classes,
                                      model_.predict(test_X))
        try:
            predicted_proba = np.append(predicted_proba,
                                        model_.predict_proba(test_X), axis=0)
        except Exception:
            tmp = np.zeros((len(test_X), no_classes), dtype=float)
            predicted_proba = np.append(predicted_proba, tmp, axis=0)
    return actual_classes, predicted_classes, predicted_proba


def classify_with_nb(final_df):
    y = final_df['isSpam']
    X = final_df.drop(['isSpam'], axis=1)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    clf = MultinomialNB()
    # accuracy = cross_val_score(clf, X, y, cv=kf, n_jobs=1)
    # print('\tAccuracy:', np.mean(accuracy), accuracy)
    # recall = cross_val_score(clf, X, y, cv=kf, scoring='recall', n_jobs=1)
    # print('\tRecall:', np.mean(recall), recall)
    # precision = cross_val_score(clf, X, y, cv=kf,
    #                             scoring='precision', n_jobs=1)
    # print('\tPrecision:', np.mean(precision), precision)
    # f1 = cross_val_score(clf, X, y, cv=kf, scoring='f1', n_jobs=1)
    # print('\tF1:', np.mean(f1), f1)
    print('Generating confusion matrix...')
    act_class, pred_class, _ = cross_val_predict(clf,
                                                 kf,
                                                 X.to_numpy(),
                                                 y.to_numpy())
    matrix = confusion_matrix(act_class, pred_class)
    print(matrix)
    accuracy = (matrix[0][0] + matrix[1][1]) / final_df.shape[0]
    precision = matrix[1][1] / (matrix[1][1] + matrix[0][1])
    recall = matrix[1][1] / (matrix[1][1]+matrix[1][0])
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    return


def main(indir):
    print('Collecting messages...')
    df = collect_data_from_messages(indir)
    df = remove_special(df)
    print('Vectorizing messages...')
    combined_df = vectorize_features(df)
    print('Generating KMeans cluster ID\'s...')
    final_df = get_kmeans_labels(combined_df)
    print('One-hot-encoding mimetype...')
    tmp = final_df['contenttype']
    cat = tmp.astype('category')
    dummies = pd.get_dummies(cat)
    final_df = pd.concat([final_df, dummies], axis='columns')
    final_df.drop(['payload', 'contenttype'], axis=1, inplace=True)
    print('Performing Multinominal Naive Bayes Classification with 5-fold CV')
    classify_with_nb(final_df)
    return


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--directory', required=True, help='Input directory')
    args = ap.parse_args()
    main(args.directory)
