# Config file for tuning of SVR model using TPOT

import numpy as np

svr_config_dict = {

    'sklearn.svm.SVR' : {
        'kernel' : ['rbf'],
        'gamma' : np.arange(0.0, 1.01, 0.05),
        'epsilon' : [1e-4, 1e-3, 1e-2, 1e-1, 1.],
        'C' : [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25., 50., 100.],
        'tol' : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    },

    'sklearn.kernel_approximation.Nystroem': {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 11)
    },

    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    }
    
}