from pysad.models.integrations.reference_window_model import ReferenceWindowModel
from pyod.models.iforest import IForest
import numpy as np # Added import for type hinting


class IForestASD(ReferenceWindowModel):
    """An Anomaly Detection Approach Based on Isolation Forest Algorithm for Streaming Data using Sliding Window :cite:`ding2013anomaly`. Note that concept drift is not implemented since it is a part of the simulation. See Algorithm 2 in "An Anomaly Detection Approach Based on Isolation Forest Algorithm for Streaming Data using Sliding Window" paper. This method is unsupervised so it is not needed to give y as parameter.

    Args:
        initial_window_X (np.float64 array of shape (num_initial_instances,num_features)): The initial window to fit for initial calibration period. We simply apply fit to these instances (Default=None).
        window_size (int): The size of the reference window and its sliding (Default=2048).
        n_estimators (int): The number of base estimators in the ensemble (Default=100).
        max_samples (int or float or str): The number of samples to draw from X to train each base estimator (Default='auto').
        contamination (float): The amount of contamination of the data set (Default=0.1).
        max_features (int or float): The number of features to draw from X to train each base estimator (Default=1.0).
        bootstrap (bool): If True, individual trees are fit on random subsets of the training data sampled with replacement (Default=False).
        n_jobs (int): The number of jobs to run in parallel for both fit and predict (Default=1).
        behaviour (str): Behaviour of the decision_function ('old' or 'new') (Default='old'). Note: 'old' is deprecated.
    """

    def __init__(self, initial_window_X: np.ndarray = None, window_size: int = 2048,
                 n_estimators: int = 100, max_samples: any = 'auto', contamination: float = 0.1,
                 max_features: float = 1.0, bootstrap: bool = False, n_jobs: int = 1,
                 behaviour: str = 'old'):
        self.initial_window_X = initial_window_X
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.behaviour = behaviour
        super().__init__(model_cls=IForest, window_size=self.window_size, sliding_size=self.window_size,
                         initial_window_X=self.initial_window_X,
                         n_estimators=self.n_estimators, max_samples=self.max_samples, contamination=self.contamination,
                         max_features=self.max_features, bootstrap=self.bootstrap, n_jobs=self.n_jobs,
                         behaviour=self.behaviour)
        # TODO: implement concept drift method