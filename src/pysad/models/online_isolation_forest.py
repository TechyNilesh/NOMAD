from __future__ import annotations

import numpy as np
from numpy import (
    argsort, asarray, empty, finfo, inf, log, ndarray, 
    sort, split, vstack, zeros
)
from numpy.linalg import norm
from numpy.random import default_rng, Generator
from typing import Callable, Literal, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from multiprocessing import cpu_count
from dataclasses import dataclass

from pysad.core.base_model import BaseModel


class OnlineIsolationForest(BaseModel):
    """Online Isolation Forest

    This class implements the Online Isolation Forest (oIFOR) algorithm, which is
    an ensemble anomaly detector capable of adapting to concept drift.

    Reference:

    `Online Isolation Forest.
    Filippo Leveni, Guilherme Weigert Cassales, Bernhard Pfahringer, Albert Bifet, and Giacomo Boracchi.
    International Conference on Machine Learning (ICML), Proceedings of Machine Learning Research (PMLR), 2024.
    <https://proceedings.mlr.press/v235/leveni24a.html>`_

    Args:
        num_trees (int): Number of trees in the ensemble. Default is 32.
        max_leaf_samples (int): Maximum number of samples per leaf. When this number is reached, a split is performed. Default is 32.
        growth_criterion (str): When to perform a split. If 'adaptive', the max_leaf_samples grows with tree depth, otherwise 'fixed'. Default is 'adaptive'.
        subsample (float): Probability of learning a new sample in each tree. Default is 1.0.
        window_size (int): The size of the window for each tree. Default is 2048.
        branching_factor (int): Branching factor of each tree. Default is 2.
        split (str): Type of split performed at each node. Currently only 'axisparallel' is supported. Default is 'axisparallel'.
        n_jobs (int): Number of parallel jobs. Default is 1.
        random_state (int): Random seed for reproducibility. Default is None.
    """

    def __init__(
        self,
        num_trees: int = 32,
        max_leaf_samples: int = 32,
        growth_criterion: Literal["fixed", "adaptive"] = "adaptive",
        subsample: float = 1.0,
        window_size: int = 2048,
        branching_factor: int = 2,
        split: Literal["axisparallel"] = "axisparallel",
        n_jobs: int = 1,
        random_state: Optional[int] = 42,
    ):
        self.num_trees = num_trees
        self.max_leaf_samples = max_leaf_samples
        self.growth_criterion = growth_criterion
        self.subsample = subsample
        self.window_size = window_size
        self.branching_factor = branching_factor
        self.split = split
        self.n_jobs = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
        self.random_state = random_state
        
        self.to_init = True
        self.random_generator = None
        self.trees = []
        self.data_window = []
        self.data_size = 0
        self.normalization_factor = 0

    def fit_partial(self, X, y=None):
        """Fits the model to next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: Returns the self.
        """
        if self.to_init:
            self._initialize(X)
            
        X = X.reshape(1, -1)
        self._learn_batch(X)
        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        if self.to_init:
            # If not initialized yet, return default score of 0.5
            return 0.5
            
        X = X.reshape(1, -1)
        return self._score_batch(X)[0]

    def _initialize(self, X):
        """Initialize the model with the first instance."""
        self.num_features = X.shape[0]
        self.random_generator = default_rng(seed=self.random_state)
        self.trees = [
            OnlineIsolationTree(
                max_leaf_samples=self.max_leaf_samples,
                growth_criterion=self.growth_criterion,
                subsample=self.subsample,
                branching_factor=self.branching_factor,
                data_size=self.data_size,
                split=self.split,
                random_seed=self.random_generator.integers(0, 2**32),
            )
            for _ in range(self.num_trees)
        ]
        self.to_init = False

    def _learn_batch(self, data: ndarray) -> OnlineIsolationForest:
        # Update the counter of data seen so far
        self.data_size += data.shape[0]
        # Compute the normalization factor
        self.normalization_factor: float = OnlineIsolationTree._get_random_path_length(
            self.branching_factor,
            self.max_leaf_samples,
            self.data_size * self.subsample,
        )
        # Instantiate a list of OnlineIsolationTrees' learn functions
        learn_funcs: list[Callable[[ndarray], OnlineIsolationForest]] = [
            tree._learn for tree in self.trees
        ]
        # OnlineIsolationTrees learn new data
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            self.trees: list[OnlineIsolationTree] = list(
                executor.map(
                    lambda f, x: f(x), learn_funcs, repeat(data, self.num_trees)
                )
            )
        # If the window size is not None, add new data to the window and eventually remove old ones
        if self.window_size:
            # Update the window of data seen so far
            self.data_window += list(data)
            # If the window size is smaller than the number of data seen so far, unlearn old data
            if self.data_size > self.window_size:
                # Extract old data and update the window of data seen so far
                data, self.data_window = (
                    asarray(self.data_window[: self.data_size - self.window_size]),
                    self.data_window[self.data_size - self.window_size :],
                )
                # Update the counter of data seen so far
                self.data_size -= self.data_size - self.window_size
                # Compute the normalization factor
                self.normalization_factor: float = (
                    OnlineIsolationTree._get_random_path_length(
                        self.branching_factor,
                        self.max_leaf_samples,
                        self.data_size * self.subsample,
                    )
                )
                # Instantiate a list of OnlineIsolationTrees' unlearn functions
                unlearn_funcs: list[Callable[[ndarray], OnlineIsolationForest]] = [
                    tree._unlearn for tree in self.trees
                ]
                # OnlineIsolationTrees unlearn new data
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    self.trees: list[OnlineIsolationTree] = list(
                        executor.map(
                            lambda f, x: f(x),
                            unlearn_funcs,
                            repeat(data, self.num_trees),
                        )
                    )
        return self

    def _score_batch(self, data: ndarray) -> ndarray:
        # Collect OnlineIsolationTrees' predict functions
        predict_funcs: list[Callable[[ndarray], OnlineIsolationForest]] = [
            tree._predict for tree in self.trees
        ]
        # Compute the depths of all samples in each tree
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            depths: ndarray = asarray(
                list(
                    executor.map(
                        lambda f, x: f(x), predict_funcs, repeat(data, self.num_trees)
                    )
                )
            ).T
        # Compute the mean depth of each sample along all trees
        mean_depths: ndarray = depths.mean(axis=1)
        # Compute normalized mean depths
        normalized_mean_depths: ndarray = 2 ** (
            -mean_depths / (self.normalization_factor + finfo(float).eps)
        )
        return normalized_mean_depths


class OnlineIsolationTree:
    def __init__(
        self,
        random_seed: int,
        max_leaf_samples: int,
        growth_criterion: Literal["fixed", "adaptive"],
        subsample: float,
        branching_factor: int,
        data_size: int,
        split: Literal["axisparallel"] = "axisparallel",
    ):
        self.random_generator: Generator = default_rng(seed=random_seed)
        self.max_leaf_samples: int = max_leaf_samples
        self.growth_criterion: Literal["fixed", "adaptive"] = growth_criterion
        self.subsample: float = subsample
        self.branching_factor: int = branching_factor
        self.data_size: int = data_size
        self.split: Literal["axisparallel"] = split
        self.depth_limit: float = OnlineIsolationTree._get_random_path_length(
            self.branching_factor,
            self.max_leaf_samples,
            self.data_size * self.subsample,
        )
        self.root: Optional[OnlineIsolationNode] = None
        self.next_node_index: int = 0

    @staticmethod
    def _get_random_path_length(
        branching_factor: int, max_leaf_samples: int, num_samples: float
    ) -> float:
        if num_samples < max_leaf_samples:
            return 0
        else:
            return log(num_samples / max_leaf_samples) / log(2 * branching_factor)

    @staticmethod
    def _get_multiplier(
        growth_criterion: Literal["fixed", "adaptive"], depth: int
    ) -> int:
        # Compute the multiplier according to the growth criterion
        if growth_criterion == "fixed":
            return 1
        elif growth_criterion == "adaptive":
            return 2**depth
        else:
            raise ValueError("Bad grow criterion {}".format(growth_criterion))

    @staticmethod
    def _split_data(
        data: ndarray, projection_vector: ndarray, split_values: ndarray
    ) -> list[ndarray]:
        # Project data using projection vector
        projected_data: ndarray = data @ projection_vector
        # Sort projected data and keep sort indices
        sort_indices: ndarray = argsort(projected_data)
        # Split data according to their membership
        partition: list[ndarray] = split(
            sort_indices, projected_data[sort_indices].searchsorted(split_values)
        )
        return partition

    def _learn(self, data: ndarray) -> OnlineIsolationTree:
        # Subsample data in order to improve diversity among trees
        data: ndarray = data[
            self.random_generator.random(data.shape[0]) < self.subsample
        ]
        if data.shape[0] >= 1:
            # Update the counter of data seen so far
            self.data_size += data.shape[0]
            # Adjust depth limit according to data seen so far and branching factor
            self.depth_limit: float = OnlineIsolationTree._get_random_path_length(
                self.branching_factor, self.max_leaf_samples, self.data_size
            )
            # Recursively update the tree
            self.next_node_index, self.root = (
                self._recursive_build(data)
                if self.root is None
                else self._recursive_learn(self.root, data, self.next_node_index)
            )
        return self

    def _recursive_learn(
        self, node: OnlineIsolationNode, data: ndarray, node_index: int
    ) -> Tuple[int, OnlineIsolationNode]:
        # Update the number of data seen so far by the current node
        node.data_size += data.shape[0]
        # Update the vectors of minimum and maximum values seen so far by the current node
        node.min_values: ndarray = vstack([data, node.min_values]).min(axis=0)
        node.max_values: ndarray = vstack([data, node.max_values]).max(axis=0)
        # If the current node is a leaf, try to split it
        if node.children is None:
            # If there are enough samples to be split according to the max leaf samples and the depth limit has not been
            # reached yet, split the node
            if (
                node.data_size
                >= self.max_leaf_samples
                * OnlineIsolationTree._get_multiplier(self.growth_criterion, node.depth)
                and node.depth < self.depth_limit
            ):
                # Sample data_size points uniformly at random within the bounding box defined by the vectors of minimum
                # and maximum values of data seen so far by the current node
                data_sampled: ndarray = self.random_generator.uniform(
                    node.min_values,
                    node.max_values,
                    size=(node.data_size, data.shape[1]),
                )
                return self._recursive_build(
                    data_sampled, depth=node.depth, node_index=node_index
                )
            else:
                return node_index, node
        # If the current node is not a leaf, recursively update all its children
        else:
            # Partition data
            partition_indices: list[ndarray] = self._split_data(
                data, node.projection_vector, node.split_values
            )
            # Recursively update children
            for i, indices in enumerate(partition_indices):
                node_index, node.children[i] = self._recursive_learn(
                    node.children[i], data[indices], node_index
                )
            return node_index, node

    def _recursive_build(
        self, data: ndarray, depth: int = 0, node_index: int = 0
    ) -> Tuple[int, OnlineIsolationNode]:
        # If there aren't enough samples to be split according to the max leaf samples or the depth limit has been
        # reached, build a leaf node
        if (
            data.shape[0]
            < self.max_leaf_samples
            * OnlineIsolationTree._get_multiplier(self.growth_criterion, depth)
            or depth >= self.depth_limit
        ):
            return node_index + 1, OnlineIsolationNode(
                data_size=data.shape[0],
                children=None,
                depth=depth,
                node_index=node_index,
                min_values=data.min(axis=0, initial=inf),
                max_values=data.max(axis=0, initial=-inf),
                projection_vector=None,
                split_values=None,
            )
        else:
            # Sample projection vector
            if self.split == "axisparallel":
                projection_vector: ndarray = zeros(data.shape[1])
                projection_vector[
                    self.random_generator.choice(projection_vector.shape[0])
                ]: float = 1.0
            else:
                raise ValueError("Bad split {}".format(self.split))
            projection_vector: ndarray = projection_vector / norm(
                projection_vector
            )
            # Project sampled data using projection vector
            projected_data: ndarray = data @ projection_vector
            # Sample split values
            split_values: ndarray = sort(
                self.random_generator.uniform(
                    min(projected_data),
                    max(projected_data),
                    size=self.branching_factor - 1,
                )
            )
            # Partition sampled data
            partition_indices: list[ndarray] = self._split_data(
                data, projection_vector, split_values
            )
            # Generate recursively children nodes
            children: ndarray = empty(
                shape=(self.branching_factor,), dtype=object
            )
            for i, indices in enumerate(partition_indices):
                node_index, children[i] = self._recursive_build(
                    data[indices], depth + 1, node_index
                )
            return node_index + 1, OnlineIsolationNode(
                data_size=data.shape[0],
                children=children,
                depth=depth,
                node_index=node_index,
                min_values=data.min(axis=0),
                max_values=data.max(axis=0),
                projection_vector=projection_vector,
                split_values=split_values,
            )

    def _unlearn(self, data: ndarray) -> OnlineIsolationTree:
        # Subsample data in order to improve diversity among trees
        data: ndarray = data[
            self.random_generator.random(data.shape[0]) < self.subsample
        ]
        if data.shape[0] >= 1:
            # Update the counter of data seen so far
            self.data_size -= data.shape[0]
            # Adjust depth limit according to data seen so far and branching factor
            self.depth_limit: float = OnlineIsolationTree._get_random_path_length(
                self.branching_factor, self.max_leaf_samples, self.data_size
            )
            # Recursively update the tree
            self.root: OnlineIsolationNode = self._recursive_unlearn(self.root, data)
        return self

    def _recursive_unlearn(
        self, node: OnlineIsolationNode, data: ndarray
    ) -> OnlineIsolationNode:
        # Update the number of data seen so far by the current node
        node.data_size -= data.shape[0]
        # If the current node is a leaf, return it
        if node.children is None:
            return node
        # If the current node is not a leaf, try to unsplit it
        else:
            # If there are not enough samples according to max leaf samples, unsplit the node
            if (
                node.data_size
                < self.max_leaf_samples
                * OnlineIsolationTree._get_multiplier(self.growth_criterion, node.depth)
            ):
                return self._recursive_unbuild(node)
            # If there are enough samples according to max leaf samples, recursively update all its children
            else:
                # Partition data
                partition_indices: list[ndarray] = self._split_data(
                    data, node.projection_vector, node.split_values
                )
                # Recursively update children
                for i, indices in enumerate(partition_indices):
                    node.children[i]: OnlineIsolationNode = self._recursive_unlearn(
                        node.children[i], data[indices]
                    )
                # Update the vectors of minimum and maximum values seen so far by the current node
                node.min_values: ndarray = vstack(
                    [node.children[i].min_values for i, _ in enumerate(node.children)]
                ).min(axis=0)
                node.max_values: ndarray = vstack(
                    [node.children[i].max_values for i, _ in enumerate(node.children)]
                ).max(axis=0)
                return node

    def _recursive_unbuild(self, node: OnlineIsolationNode) -> OnlineIsolationNode:
        # If the current node is a leaf, return it
        if node.children is None:
            return node
        # If the current node is not a leaf, unbuild it
        else:
            # Recursively unbuild children
            for i, _ in enumerate(node.children):
                node.children[i]: OnlineIsolationNode = self._recursive_unbuild(
                    node.children[i]
                )
            # Update the vectors of minimum and maximum values seen so far by the current node
            node.min_values: ndarray = vstack(
                [node.children[i].min_values for i, _ in enumerate(node.children)]
            ).min(axis=0)
            node.max_values: ndarray = vstack(
                [node.children[i].max_values for i, _ in enumerate(node.children)]
            ).max(axis=0)
            # Delete children nodes, projection vector and split values
            node.children: Optional[ndarray] = None
            node.projection_vector: Optional[ndarray] = None
            node.split_values: Optional[ndarray] = None
            return node

    def _predict(self, data: ndarray) -> ndarray:
        # Compute depth of each sample
        if self.root:
            return self._recursive_depth_search(
                self.root, data, empty(shape=(data.shape[0],), dtype=float)
            )
        else:
            return zeros(shape=(data.shape[0],), dtype=float)

    def _recursive_depth_search(
        self, node: OnlineIsolationNode, data: ndarray, depths: ndarray
    ) -> ndarray:
        # If the current node is a leaf, fill the depths vector with the current depth plus a normalization factor
        if node.children is None or data.shape[0] == 0:
            depths[:] = node.depth + OnlineIsolationTree._get_random_path_length(
                self.branching_factor, self.max_leaf_samples, node.data_size
            )
        else:
            # Partition data
            partition_indices: list[ndarray] = self._split_data(
                data, node.projection_vector, node.split_values
            )
            # Fill the vector of depths
            for i, indices in enumerate(partition_indices):
                depths[indices]: ndarray = self._recursive_depth_search(
                    node.children[i], data[indices], depths[indices]
                )
        return depths


@dataclass
class OnlineIsolationNode:
    def __init__(
        self,
        data_size: int,
        children: Optional[ndarray],
        depth: int,
        node_index: int,
        min_values: ndarray,
        max_values: ndarray,
        projection_vector: Optional[ndarray],
        split_values: Optional[ndarray],
    ):
        self.data_size: int = data_size
        self.children: Optional[ndarray] = children
        self.depth: int = depth
        self.node_index: int = node_index
        self.min_values: ndarray = min_values
        self.max_values: ndarray = max_values
        self.projection_vector: Optional[ndarray] = projection_vector
        self.split_values: Optional[ndarray] = split_values