from collections import Counter

from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from tscluster.tskmeans import TSKmeans

from codes.utils import *

def create_design_data(X, maxlag, y_col_idx, fit_intercept):
    F = X.shape[1]

    feature_col_mask = np.array([True] * F)
    feature_col_mask[y_col_idx] = False 
    
    
    Xi = X[:, feature_col_mask].copy()
    Xi_u = create_lag_features(Xi[:-1, :], maxlag, add_constant=fit_intercept)
    Xi_r = create_lag_features(X[:-1, [y_col_idx]].copy(), maxlag, add_constant=fit_intercept)

    if fit_intercept:
        Xi_u = np.hstack([Xi_r[:, :-1], Xi_u]) # ensure that the ones for the constant coef is included once.
    else:
        Xi_u = np.hstack([Xi_r, Xi_u])

    yi = X[maxlag:, [y_col_idx]].copy()

    return Xi_u, Xi_r, yi

def cal_local_lr_theta(cov_x_lst, cov_xy_lst):
    return np.matmul(np.linalg.inv(np.sum(np.array(cov_x_lst), axis=0)), np.sum(np.array(cov_xy_lst), axis=0))

class Base_PTSC():

    @property 
    def labels_(self):
        return self._labels
    
    @property 
    def important_feature_set_(self):
        return self._imp_features

class ARTSC(Base_PTSC):
    def __init__(
            self, 
            k, 
            maxlag=1, 
            fit_intercept=True, 
            metric='euclidean', 
            alpha=0.1,
            random_state=None
            ):
        
        super().__init__()

        self.k = k
        self.maxlag = maxlag
        self.fit_intercept = fit_intercept
        self.metric = metric
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X):

        ts_kmeans = TSKmeans(n_clusters=self.k, metric=self.metric, random_state=self.random_state)

        ts_kmeans.fit(X)
        labels = ts_kmeans.labels_
        
        _, N, F = X.shape

        theta_us = []

        for i in range(self.k):

            Xi = ts_kmeans.cluster_centers_[:, i, :]

            Xi_u, _, yi = create_design_data(Xi, maxlag=self.maxlag, y_col_idx=-1, fit_intercept=False) # sklearn's regression has coef_ and intercept_ attributes 
                                                                                                # for coef and intercept

            lasso_reg = Lasso(alpha=self.alpha, fit_intercept=self.fit_intercept) 
            lasso_reg.fit(Xi_u, yi)

            theta = lasso_reg.coef_ # sklearn's regression has coef_ and intercept_ attributes for coef and intercept

            # end_idx = -1 if fit_intercept else len(theta)

            theta = set(np.array([np.arange(0, F-1)] * self.maxlag).T.flatten()[np.abs(theta[self.maxlag:]) > 0]) # empty set could mean only the lag values of  
                                                                                                        # the target time series are important

            if len(theta) == 0:
                theta = {-1}

            theta = tuple(sorted(list(theta)))

            theta_us.append(theta)

        theta_us = dict(enumerate(theta_us))

        self._labels, self._imp_features = labels, theta_us

        return self

class KMW(Base_PTSC):
    def __init__(
            self, 
            k, 
            maxlag=1, 
            fit_intercept=True, 
            alpha=0.1,
            random_state=None
            ):
        
        super().__init__()

        self.k = k
        self.maxlag = maxlag
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X):
        _, N, F = X.shape

        theta_us = []

        X_u_lst = []
        y_lst = []

        for i in range(N):
            cov_x_u_lst = []
            cov_xy_u_lst = []

            Xi = X[:, i, :]

            Xi_u, _, yi = create_design_data(Xi, maxlag=self.maxlag, y_col_idx=-1, fit_intercept=self.fit_intercept)

            Xi_u_T = Xi_u.T

            cov_x_u_lst.append(np.matmul(Xi_u_T, Xi_u))
            cov_xy_u_lst.append(np.matmul(Xi_u_T, yi))

            theta_us.append(cal_local_lr_theta(cov_x_u_lst, cov_xy_u_lst).flatten())

            X_u_lst.append(Xi_u)
            y_lst.append(yi)

        theta_us = np.array(theta_us)

        if self.fit_intercept:
            theta_us = theta_us[:, :-1]

        theta_us = theta_us[:, self.maxlag:] # removing target feature

        km_u = KMeans(self.k, random_state=self.random_state)

        theta_us = np.abs(theta_us)

        km_u.fit(theta_us) 

        theta_u = km_u.cluster_centers_ 

        theta_bool = theta_u > np.mean(theta_u, axis=1).reshape(-1, 1)
        feature_lag_lst = np.array(sum([[i]*self.maxlag for i in range(F-1)], []))

        theta_u = {k: tuple(np.unique(feature_lag_lst[theta_bool[k]])) for k in range(theta_bool.shape[0])}
        
        labels = km_u.labels_ 

        self._labels, self._imp_features = labels, theta_u

        return self

class AggLR(Base_PTSC):

    def __init__(
            self, 
            k, 
            maxlag=1, 
            fit_intercept=True, 
            alpha=0.1
            ):
        
        super().__init__()

        self.k = k
        self.maxlag = maxlag
        self.fit_intercept = fit_intercept
        self.alpha = alpha

    def fit(self, X):
        _, N, F = X.shape

        theta_us = []

        for i in range(N):

            Xi = X[:, i, :]

            Xi_u, _, yi = create_design_data(Xi, maxlag=self.maxlag, y_col_idx=-1, fit_intercept=False) # sklearn's regression has coef_ and intercept_ 
                                                                                                # attributes for coef and intercept

            lasso = Lasso(alpha=self.alpha, fit_intercept=self.fit_intercept)
            lasso.fit(Xi_u, yi)

            theta = lasso.coef_
            theta = set(np.array([np.arange(0, F-1)] * self.maxlag).T.flatten()[np.abs(theta[self.maxlag:]) > 0]) # empty set could mean only the lag values of  
                                                                                                    # the target time series are important

            if len(theta) == 0: # empty set could mean only the lag values of the target time series are important, so we use -1 as the class
                theta = {-1}

            theta = ' '.join(list(map(str, sorted(list(theta)))))
            
            theta_us.append(theta)

        vectorizer = CountVectorizer(binary=True, tokenizer=lambda x: x.split())

        # Fit and transform the data
        one_hot_encoded = vectorizer.fit_transform(theta_us).toarray()
        feature_names = vectorizer.get_feature_names_out()

        jaccard_dist = pairwise_distances(one_hot_encoded, metric='jaccard')

        # Perform Agglomerative Clustering with Jaccard distance
        agg_clust = AgglomerativeClustering(n_clusters=self.k, metric='precomputed', linkage='average')

        agg_clust.fit(jaccard_dist)

        labels = agg_clust.labels_

        # most common important feature set is the class for a cluster
        theta_us = []
        for j in range(self.k):
            list_of_feature_set = list(map(lambda row: ' '.join(feature_names[row == 1]), one_hot_encoded[labels == j, :]))

            counter = Counter(list_of_feature_set)
            theta_us.append(tuple(sorted(map(int, list(set(counter.most_common(1)[0][0].split()))))))

        theta_us = dict(enumerate(theta_us))

        self._labels, self._imp_features = labels, theta_us

        return self