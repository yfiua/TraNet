from __future__ import division

import sys
import igraph
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
from transform_funcs import *
from utils import *
from scipy.sparse import coo_matrix, linalg

def graph_to_sparse_matrix(G):
    n = G.vcount()
    xs, ys = map(np.array, zip(*G.get_edgelist()))
    if not G.is_directed():
        xs, ys = np.hstack((xs, ys)).T, np.hstack((ys, xs)).T
    else:
        xs, ys = xs.T, ys.T
    return coo_matrix((np.ones(xs.shape), (xs, ys)), shape=(n, n), dtype=np.int16)

def get_feature(G, f):
    return _transform_func_degree(getattr(G, f)()) if callable(getattr(G, f)) else _transform_func(getattr(G, f))

# aggregate by the mean value of feature of neighbours
def mean_neighbour(A, d, feature):
    return A.dot(feature) / d

def get_feature_matrix(G, features, rounds=5):
    # local clustering coefficient
    G_sim = G.as_directed().simplify(multiple=False)                 # remove loops

    lcc = np.array(G_sim.transitivity_local_undirected(mode='zero'))
    lcc[lcc < 0] = 0                                                 # implementation of igraph is really shitty
    G.clustering_coefficient = lcc

    # compute PageRank
    G_sim = G.copy()
    G_sim = G_sim.simplify(multiple=False)                           # remove loops

    alpha = 0.15
    pagerank = np.array(G_sim.pagerank(damping=1-alpha))
    G.pr = pagerank

    feature_matrix = [ get_feature(G, f) for f in features ]
    X = np.array(feature_matrix).T

    # adjacency matrix (simplified)
    A = graph_to_sparse_matrix(G.as_undirected().simplify())
    d = np.squeeze(np.array(A.sum(axis=1))).astype(np.int)
    d[d == 0] = 1

    for i in range(rounds):
        feature_matrix = [ mean_neighbour(A, d, f) for f in feature_matrix ]
        X = np.concatenate((X, np.array(feature_matrix).T), axis=1)

    #X = np.hstack((X, np.array([pagerank]).T))
    return X

def read_data(lang, features):
    # dataset (network)
    df = pd.read_csv('data/' + lang + '-wiki-talk', sep='\t', header=None)

    nodes = np.unique(df[[0, 1]].values);
    max_node_num = max(nodes) + 1
    num_nodes = len(nodes)

    G = igraph.Graph(directed=True)
    G.add_vertices(max_node_num)
    G.add_edges(df[[0, 1]].values)

    G = G.subgraph(nodes)

    # features
    X = get_feature_matrix(G, features)

    # dataset (roles)
    df_role = pd.read_csv('data/' + lang + '-user-group', sep='\t', header=None)
    roles = df_role[[0,1]].values

    y = [0] * max_node_num
    for r in roles:
        y[r[0]] = r[1]

    y = np.array([y[i] for i in nodes])

    return np.squeeze(X), y

# main
def main():
    # params
    n_trees = 200
    features = [ 'clustering_coefficient' , 'degree' , 'indegree' , 'outdegree', 'pr' ]
    langs = [ 'ar', 'bn', 'br', 'ca', 'cy', 'de', 'el' , 'en', 'eo', 'es', 'eu', 'fr', 'gl', 'ht', 'it', 'ja', 'lv', 'nds', 'nl', 'oc', 'pl', 'pt', 'ru', 'sk', 'sr', 'sv', 'vi', 'zh' ]
    #langs = [ 'br', 'cy', 'ar', 'lv', 'zh' ]

    # read datasets
    X = {}
    y = {}
    for lang in langs:
        X[lang], y[lang] = read_data(lang, features)

    # admin classifier
    for lang_source in langs:
        y_source = (y[lang_source] == 2)

        ## classifier
        clf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        clf.fit(X[lang_source], y_source)

        ## evaluation
        for lang_target in langs:
            y_target = (y[lang_target] == 2)
            if (len(np.unique(y_source)) == 1) or (len(np.unique(y_target)) == 1):   # ROC not defined
                auc = np.nan
            else:
                y_predict = clf.predict_proba(X[lang_target])[:,1]
                auc = roc_auc_score(y_target, y_predict)

            print lang_source + ',' + lang_target + ',' + str(auc)

if __name__ == '__main__':
    # init
    _transform_func_degree = no_transform
    _transform_func = no_transform

    main()
