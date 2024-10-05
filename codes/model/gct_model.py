from itertools import combinations
from functools import reduce

import scipy.stats as stats
from sklearn.cluster import KMeans

from codes.utils import *

class GrangeCausalTree():

    def __init__(self, 
                 k=None, 
                 maxlag=1, 
                 fit_intercept=True, 
                 lambda_reg=0, 
                 p_value_threshold=None
                 ):
        
        self.k = k
        self.maxlag = maxlag
        self.fit_intercept = fit_intercept
        self.lambda_reg = lambda_reg
        self.p_value_threshold = p_value_threshold

    def create_design_data_GC(self, X, x_col_idx):

        Xi = X[:-1, x_col_idx].copy()
        return create_lag_features(Xi, self.maxlag, add_constant=self.fit_intercept)
        
    def get_likelihoods(self, ts_arr, feature_set, which='both', return_thetas=False):

        log_likelihoods_u = np.zeros((ts_arr.shape[1], len(feature_set)))
        log_likelihoods_r = np.zeros(ts_arr.shape[1])

        for i in range(ts_arr.shape[1]):
            X = ts_arr[:, i, :]
            
            # scaler = MinMaxScaler()
            # X = scaler.fit_transform(X)
            
            y = X[self.maxlag:, [-1]].copy()

            n = X.shape[0]

            if which in ('both', 'unrestricted'):
                for m in range(len(feature_set)):
                    p = list(feature_set[m]) + [-1]

                    # mask = np.ones(X.shape[1], dtype=bool)
                    # mask[list(p)] = False

                    X_u = self.create_design_data_GC(X, x_col_idx=p)

                    b_u = np.matmul(np.linalg.inv(np.matmul(X_u.T, X_u) + self.lambda_reg * np.identity(X_u.shape[1])), 
                                    np.matmul(X_u.T, y))                    

                    rss_u = np.sum((y - np.matmul(X_u, b_u))**2) + self.lambda_reg * np.linalg.norm(b_u.flatten())**2
                    
                    log_likelihoods_u[i, m] = n * np.log(rss_u) 

            # if which in ('both', 'restricted'):
            X_r = self.create_design_data_GC(X, x_col_idx=[-1])

            b_r = np.matmul(np.linalg.inv(np.matmul(X_r.T, X_r) + self.lambda_reg * np.identity(X_r.shape[1])), 
                            np.matmul(X_r.T, y))

            rss_r = np.sum((y - np.matmul(X_r, b_r))**2) + self.lambda_reg * np.linalg.norm(b_r.flatten())**2
            
            log_likelihoods_r[i] = n * np.log(rss_r) 

        if which == 'unrestricted':
            if return_thetas and len(feature_set) == 1 and ts_arr.shape[1] == 1:
                return log_likelihoods_r[0] - log_likelihoods_u[0], b_u.flatten()

            return log_likelihoods_u
    
        elif which == 'restricted':
            return log_likelihoods_r 
        
        return log_likelihoods_u, log_likelihoods_r
        
    def build_tree(self, i, arr, log_likelihoods_r, feature_set_dict, nodes, nodes_to_skip=[], verbose=False):
        
        df = self.maxlag
        
        log_likelihoods_u_i = np.zeros(len(feature_set_dict)).astype(float)
        log_likelihoods_u_i[:] = np.nan

        path = []

        min_p_value = np.inf

        for depth_i in range(len(nodes)):
            depth = nodes[depth_i]
            p_values = []
            node_lst = []            

            for node_i in range(len(depth)):
                node = depth[node_i]

                if node in nodes_to_skip:
                    continue
            
                if depth_i == 0:
                    log_likelihood_r = log_likelihoods_r[i]

                    log_likelihood_u = self.get_likelihoods(arr, [node], which='unrestricted')[0, 0]
                    log_likelihoods_u_i[feature_set_dict[node]] = log_likelihood_u

                    likelihood_ratio = log_likelihood_r - log_likelihood_u
                    p_value = 1 - stats.chi2.cdf(likelihood_ratio, df)

                    p_values.append(p_value)
                    node_lst.append(node)
                    
                    if verbose:
                        print(f"depth: {depth}, node: {node}, p_value: {p_value}, lr: {likelihood_ratio}")
                
                else:

                    prev_node = path[-1]

                    if len(set(node) - set(prev_node)) == 1:

                        log_likelihood_r = log_likelihoods_u_i[feature_set_dict[prev_node]]
                    else:
                        continue
                        
                    log_likelihood_u = self.get_likelihoods(arr, [node], which='unrestricted')[0, 0]
                    log_likelihoods_u_i[feature_set_dict[node]] = log_likelihood_u
                    
                    likelihood_ratio = log_likelihood_r - log_likelihood_u
                    p_value = 1 - stats.chi2.cdf(likelihood_ratio, df)

                    # if prev_node == path[-1]:
                    p_values.append(p_value)
                    # prev_node_lst.append(prev_node)
                    node_lst.append(node)

                    if verbose:
                        print(f"depth: {depth}, node: {node}, prev_node: {prev_node}, p_value: {p_value}, lr: {likelihood_ratio}")

            min_idx = np.argmin(p_values)


            if depth_i == 0 or (self.p_value_threshold is None and p_values[min_idx] < min_p_value):
                path.append(node_lst[min_idx])
                min_p_value = p_values[min_idx]

                if verbose:
                    print(path)

            elif self.p_value_threshold is not None: 
                if p_values[min_idx] < self.p_value_threshold: # here because we can't do float < None
                    path.append(node_lst[min_idx])

                else: # if p_value is not less than threshold
                    break 

                if verbose:
                    print(path)

            else: # break if threshold is None and p_value not less than min_p_value
                break
            
        return reduce(lambda x, y: x | y, map(set, path)), log_likelihoods_u_i

    def GC_PTSC(self, ts_arr, verbose=False, use_agg_clustering=False):
        
        N = ts_arr.shape[1]
            
        n_features = ts_arr.shape[2] - 1
        feature_lst = sum([list(combinations(np.arange(n_features), i)) for i in range(1, n_features+1)], [])

        feature_lst_dict = {v: k for k, v in enumerate(feature_lst)}

        nodes = [[p for p in feature_lst if len(p) == i] for i in range(1, n_features + 1)]

        labels = np.zeros(N)
        labels[:] = -1

        important_features = {}

        log_likelihoods_r = self.get_likelihoods(ts_arr, feature_lst, which='restricted')
        log_likelihoods_u = np.zeros((ts_arr.shape[1], len(feature_lst))).astype(float)
        log_likelihoods_u[:, :] = np.nan
        
        # max_lr_matrix = np.zeros(log_likelihoods_u.shape)
        for i in range(N):
            # important_feature_set, max_lr_arr = build_tree(i, log_likelihoods_u, log_likelihoods_r, feature_lst_dict, nodes, maxlag, 
            #                                                      return_max_lr_arr=True, verbose=verbose)

            important_feature_set, log_likelihoods_u_i = self.build_tree(i, ts_arr[:, [i], :], log_likelihoods_r, 
                                                                feature_lst_dict, nodes, verbose=verbose
                                                                )
            
            log_likelihoods_u[i, :] = log_likelihoods_u_i

            important_feature_set = tuple(sorted(list(important_feature_set)))

            if important_feature_set not in important_features:
                important_features[important_feature_set] = int(np.max(labels) + 1)

            labels[i] = important_features[important_feature_set]

            # max_lr_matrix[i, :] = max_lr_arr

        flip_keys_and_values = lambda d: {v: k for k, v in d.items()}
        important_features = flip_keys_and_values(important_features)

        self.__n_trained_models = log_likelihoods_r.size + np.sum(~np.isnan(log_likelihoods_u)), log_likelihoods_u.size + log_likelihoods_r.size

        if self.k is None or len(important_features) == self.k: # check if there is need to recluster
            
            return labels, important_features
        
        elif use_agg_clustering:
            item_sets = list(map(lambda x: important_features[x], labels))

            thetas = np.zeros((N, ts_arr.shape[2]-1))
            for i in range(N):
                node = item_sets[i]
                _, theta = self.get_likelihoods(ts_arr[:, [i], :], [node], which='unrestricted',
                                return_thetas=True)
                # print(i, node, theta)
                
                thetas[i, list(node)] = np.max(np.abs(theta[: len(node)*self.maxlag]).reshape(-1, self.maxlag), axis=1)

            thetas = np.array(thetas)
            km = KMeans(n_clusters=self.k)
            # km.fit(one_hot_encoded)
            km.fit(thetas)
            labels = km.labels_
            return labels, important_features

        # recluster entities in clusters with low number of entities
        cluster_idx, n_samples_per_cluster = np.unique(labels, return_counts=True) # get entity count of clusters

        sorted_cluster_idx = cluster_idx[np.argsort(n_samples_per_cluster)[::-1]] # sort clusters in descending order of entity count

        clusters = sorted_cluster_idx[:self.k].copy() # clusters to choose
        nodes_idx_to_keep =  [feature_lst_dict[f] for f in [important_features[j] for j in clusters]] # nodes index to keep
        entities_2_recluster = [i for i in range(N) if labels[i] not in clusters] # enitites to recluster

        # remove clusters with low entity count from the dict
        for j in sorted_cluster_idx[self.k:]:
            important_features.pop(j)

        important_features = flip_keys_and_values(important_features)

        # get new clusters of the entities without cluster
        for i in entities_2_recluster:
            # important_feature_set = feature_lst[nodes_idx_to_keep[np.argmax(max_lr_matrix[i, nodes_idx_to_keep])]]

            for j in nodes_idx_to_keep:
                if np.isnan(log_likelihoods_u[i, j]):
                    log_likelihoods_u[i, j] = self.get_likelihoods(ts_arr[:, [i], :], [feature_lst[j]], 
                                                                   which='unrestricted')[0, 0]
                
            important_feature_set = feature_lst[nodes_idx_to_keep[np.argmax([
                (log_likelihoods_r[i] - log_likelihoods_u[i, j]) / len(feature_lst[j]) for j in nodes_idx_to_keep])]]

            labels[i] = important_features[important_feature_set]

        # renumber cluster index to be in range(k)
        important_features = flip_keys_and_values(important_features)
        new_cluster_labels = sorted(important_features.keys())

        relabeled_important_features = {}
        relabeled_labels = np.zeros(N)
        relabeled_labels[:] = np.nan

        for j, v in enumerate(new_cluster_labels):
            relabeled_important_features[j] = important_features[v]
            relabeled_labels[labels == v] = j 

        self.__n_trained_models = log_likelihoods_r.size + np.sum(~np.isnan(log_likelihoods_u)), log_likelihoods_u.size + log_likelihoods_r.size
        
        return relabeled_labels, relabeled_important_features
    
    def fit(self, X):
        self.__labels, self.__imp_features = self.GC_PTSC(X)

        return self


    @property 
    def labels_(self):
        return self.__labels
    
    @property 
    def important_feature_set_(self):
        return self.__imp_features
    
    @property
    def n_trained_models_(self):
        return self.__n_trained_models

if __name__ == "__main__":
    pass