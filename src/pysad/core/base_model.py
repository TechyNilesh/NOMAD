from abc import ABC, abstractmethod
from pysad.utils import _iterate
import numpy as np
import inspect

class BaseModel(ABC):
    """Abstract base class for the models.
    """

    @abstractmethod
    def fit_partial(self, X, y=None):
        """Fits the model to next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): The label of the instance (Optional for unsupervised models, default=None).

        Returns:
            object: Returns the self.
        """
        pass

    @abstractmethod
    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        pass

    def fit_score_partial(self, X, y=None):
        """Applies fit_partial and score_partial to the next instance, respectively.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit and score.
            y (int): The label of the instance (Optional for unsupervised models, default=None).

        Returns:
            float: The anomalousness score of the input instance.
        """
        return self.fit_partial(X, y).score_partial(X)

    def fit(self, X, y=None):
        """Fits the model to all instances in order.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances in order to fit.
            y (int): The labels of the instances in order to fit (Optional for unsupervised models, default=None).

        Returns:
            object: Fitted model.
        """
        for xi, yi in _iterate(X, y):
            self.fit_partial(xi, yi)

        return self

    def score(self, X):
        """Scores all instaces via score_partial iteratively.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances in order to score.

        Returns:
            np.float64 array of shape (num_instances,): The anomalousness scores of the instances in order.
        """
        y_pred = np.empty(X.shape[0], dtype=np.float64)
        for i, (xi, _) in enumerate(_iterate(X)):
            y_pred[i] = self.score_partial(xi)

        return y_pred

    def fit_score(self, X, y=None):
        """This helper method applies fit_score_partial to all instances in order.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances in order to fit.
            y (np.int32 array of shape (num_instances, )): The labels of the instances in order to fit (Optional for unsupervised models, default=None).

        Returns:
            np.float64 array of shape (num_instances,): The anomalousness scores of the instances in order.
        """
        y_pred = np.zeros(X.shape[0], dtype=np.float64)
        for i, (xi, yi) in enumerate(_iterate(X, y)):
            y_pred[i] = self.fit_score_partial(xi, yi)

        return y_pred
    
    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the model.
        
        Returns:
            list: Parameter names in the constructor.
        """
        # Fetch the constructor
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # Introspect the constructor arguments
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        
        # Extract and sort argument names
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this model.

        Args:
            deep (bool, default=True): If True, will return the parameters for this model
                and contained subobjects that are estimators.

        Returns:
            dict: Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
                if deep and hasattr(value, 'get_params') and not isinstance(value, type):
                    deep_items = value.get_params().items()
                    out.update((key + '__' + k, val) for k, val in deep_items)
                out[key] = value
            except AttributeError:
                # Some parameters might not be set as attributes
                pass
        return out

    def set_params(self, **params):
        """Set the parameters of this model.

        Args:
            **params: Estimator parameters.

        Returns:
            self: Estimator instance.
        """
        if not params:
            # Simple optimization for no-op case
            return self
        
        valid_params = self.get_params(deep=True)
        
        nested_params = {}
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key} for model {self}. "
                    f"Valid parameters are: {sorted(valid_params.keys())}"
                )
                
            if delim:
                if key not in nested_params:
                    nested_params[key] = {}
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value
                
        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)
            
        return self
