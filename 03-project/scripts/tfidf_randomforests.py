import csv
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter


from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering



with open("/Users/john/Dropbox/Downloaeded SAP2/trainingandtestdata/training.1600000.processed.noemoticon.csv", encoding='utf-8') as f:
    data = list(csv.reader(f))

data2 = [[x[0],x[-1]] for x in data]

#standarize capitatlizion


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words = 'english', max_df=.95, min_df=.001)
X = vectorizer.fittransform([x[1] for x in data2])
#vectorizer.vocabulary
#np.where(X.sum(axis=1)==0)[0].shape
Y = [0 if x[0]=='0' else 1 for x in data2]

results = dict()
model_store = dict()



def run_cluster(clusterMethod=KMeans, cluster_id='km', cluster_range=[5,10,15]):
    results[cluster_id]=dict()
    results[cluster_id]['n_clusters']=dict()
    model_store[cluster_id]=dict()
    model_store[cluster_id]['n_clusters']=dict()
    for x in cluster_range:
        print(x)
        sc = clusterMethod(n_clusters=x).fit(X.toarray())
        distances = sc.transform(X)
        closest_centroid_index = np.argmin(distances, axis=1)
        closest_distance = np.min(distances, axis=1)
        avg_distance = np.mean(closest_distance)
        results[cluster_id]['n_clusters']
        results[cluster_id]['n_clusters'][x]=avg_distance
        model_store[cluster_id]['n_clusters'][x]=sc

run_cluster()


from sklearn.ensemble import RandomForestClassifier
def run_clf(clf=RandomForestClassifier, X, Y):
    clf = clf()
    clf.fit(X,Y)
    clf.score(X,Y)