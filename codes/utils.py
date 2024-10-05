import numpy as np
from sklearn.preprocessing import StandardScaler

from tscluster.preprocessing.utils import ntf_to_tnf

def scale_per_entity(X):
    ar = []
    for i in range(X.shape[1]):
        sc = StandardScaler()
        ar.append(sc.fit_transform(X[:, i, :]))

    return ntf_to_tnf(np.array(ar))

def generate_uni_ts(t, noise_sigma, ar_params, intercept=0, random_state=None):
    rand_gen = np.random.RandomState(random_state)

    p = len(ar_params)

    expectation = intercept / (1 - np.sum(ar_params))

    # ar_params = np.array(list(ar_params) + [intercept])

    y = np.zeros(t)

    noise_mean = 0
    noise = rand_gen.normal(loc=noise_mean, scale=noise_sigma, size=t-p)
        
    y[:p] = rand_gen.uniform(low=expectation-1, high=1, size=p) # rand_gen.randn(p) #
    
    for t in range(p, t):
        x = list(y[t-p: t][::-1]) #+ [1]
        y[t] = np.dot(ar_params, x) + noise[t-p]
    
    y += expectation
    
    return y


def gen_synthetic_features(T=20, N=30, F=2, feature_noise_sigma=1.0, n_lags=2):

    ts_arr = np.zeros((T, N, F+1))

    ar_pars = []
    intercepts = []

    for i in range(N):
        rand_gen = np.random.RandomState(i)

        # high params for lag 1
        # ar_params_neg_1 = rand_gen.uniform(low=-0.9, high=-0.5, size=(1, F))
        # ar_params_pos_1 = rand_gen.uniform(low=0.6, high=0.9, size=(1, F))

        ar_params_neg_1 = rand_gen.uniform(low=0, high=0, size=(1, F))
        ar_params_pos_1 = rand_gen.uniform(low=0, high=0, size=(1, F))

        ar_params_1 = np.vstack([ar_params_neg_1, ar_params_pos_1])

        # low params for lag 2
        # ar_params_neg_2 = rand_gen.uniform(low=-0.2, high=-0.05, size=(1, F))
        # ar_params_pos_2 = rand_gen.uniform(low=0.05, high=0.2, size=(1, F))

        ar_params_neg_2 = rand_gen.uniform(low=0, high=0, size=(1, F))
        ar_params_pos_2 = rand_gen.uniform(low=0, high=0, size=(1, F))

        ar_params_2 = np.vstack([ar_params_neg_2, ar_params_pos_2])

        intercept = rand_gen.uniform(low=-1, high=1, size=F) #np.zeros(F)
        intercepts.append(intercept)

        ar_par = []
        for f in range(F):
            seed = i**2 + (f+1)*100

            params = [rand_gen.choice(ar_params_1[:, f], 1)[0], rand_gen.choice(ar_params_2[:, f], 1)[0]]
            
            ar_par.append(params)

            ts_arr[:, i, f] = generate_uni_ts(T, feature_noise_sigma, params, intercept=intercept[f], random_state=seed)

        ar_pars.append(ar_par)

    return ts_arr

# creating a function to create features from lags of multivariate time series
def create_lag_features(arr, n_lags, add_constant):        
    X = []
    
    for i in range(arr.shape[0] - n_lags + 1):
        if add_constant:
            lst = list(arr[i: i+n_lags].T[:, ::-1].flatten()) + [1] # + [1] for constant coefficient
        else:
            lst = arr[i: i+n_lags].T[:, ::-1].flatten()

        X.append(lst) 
            
    return np.array(X)

def sample_par_val(is_gc, rand_gen, std=0.2):
    low, high = 0, 0.4

    if is_gc:
        low, high = 0.8, 1.5
    
    avg = rand_gen.uniform(low=low, high=high, size=1)

    return rand_gen.normal(loc=avg, scale=std, size=1)[0] * rand_gen.choice([-1, 1])

def get_index_to_be_gc(n_lags, rand_gen):
    lag_idx = np.arange(n_lags)
    n = rand_gen.choice(lag_idx, size=1) + 1

    lag_idx[:] = False
    lag_idx[rand_gen.choice(np.arange(n_lags), size=n, replace=False)] = True # sample without replacement

    return lag_idx


def gen_synthetic_data(K=3, T=20, N=60, F=3, n_lags=2, target_noise_sigma=0.2, 
                       feature_noise_sigma=1.0, par_noise_std=0.1, 
                       true_set={0: (0, 1), 1: (0,), 2: (2,)}, entities_per_cluster=20, random_state=None, is_target_gc=0
                       ):
    
    ts_arr = gen_synthetic_features(T=T, N=N, F=F, feature_noise_sigma=feature_noise_sigma)

    rand_gen = np.random.RandomState(random_state)

    for k in range(K):
        for idx in range(entities_per_cluster):
            i = idx + (k*entities_per_cluster)

            ts_arr[:n_lags, i, -1] = rand_gen.uniform(low=-1, high=1, size=n_lags)
                                
            is_gc_params = np.zeros((F+1, n_lags)).astype(bool)
            
            for f in true_set[k]:
                is_gc_params[f, :] = get_index_to_be_gc(n_lags, rand_gen=rand_gen)

            if is_target_gc and rand_gen.choice([True, False]):
                is_gc_params[-1, :] = get_index_to_be_gc(n_lags, rand_gen=rand_gen)

            is_gc_params = is_gc_params.flatten()

            ar_params = np.array(list(map(lambda x: sample_par_val(x, rand_gen, par_noise_std), is_gc_params)))

            intercepts = rand_gen.uniform(low=-2, high=2, size=1)
            ar_params = np.concatenate([ar_params, intercepts])

            for t in range(n_lags, T):    
                X = create_lag_features(ts_arr[t - n_lags: t, i, :], n_lags=n_lags, add_constant=True).flatten()           

                noise = rand_gen.normal(loc=0, scale=target_noise_sigma, size=1)

                ts_arr[t, i, -1] = np.dot(X, ar_params) + noise
                
    return ts_arr

if __name__ == "__main__":
    pass
