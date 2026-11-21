import math
import random
import numpy as np
from pysad.core.base_model import BaseModel


class SiLeaf:
    """Leaf node of a Streaming Isolation Tree."""

    def __init__(self, X, up, side):
        self.instances = list(X)
        self.up = up
        self.side = side

    def walk(self, x):
        yield self

    def __repr__(self):
        return str(f"{self.mass}@{self.depth}")

    @property
    def n_nodes(self):
        return 1

    @property
    def n_branches(self):
        return 0

    @property
    def n_leaves(self):
        return 1

    @property
    def height(self):
        return 1

    @property
    def mass(self):
        return len(self.instances)

    @property
    def depth(self):
        return 1 + self.up.depth if self.up is not None else 0

    def remove_instance(self, idx):
        """Remove instance by index."""
        if idx < len(self.instances):
            self.instances.pop(idx)

    def insert_instance(self, x):
        self.instances.append(x)


class SiBranch:
    """Branch node of a Streaming Isolation Tree."""

    ROOT = "root"
    LEFT = "left"
    RIGHT = "right"

    def __init__(self, X, left, right, feature, split_value, up, side, f_min, f_max):
        self.children = [left, right]
        self.feature = feature
        self.split_value = split_value
        self.up = up
        self.side = side
        self.feature_min = f_min
        self.feature_max = f_max
        self.instances = list(X)

    @property
    def repr_split(self):
        return f"{self.feature} < {self.split_value:.5f}"

    def walk(self, x):
        """Iterate over the nodes of the path induced by x."""
        yield self
        yield from self.next(x).walk(x)

    def traverse(self, x):
        """Return the leaf corresponding to the given input."""
        for node in self.walk(x):
            pass
        return node

    @property
    def n_nodes(self):
        """Number of descendants, including thyself."""
        return 1 + sum(child.n_nodes for child in self.children)

    @property
    def n_branches(self):
        """Number of branches, including thyself."""
        return 1 + sum(child.n_branches for child in self.children)

    @property
    def n_leaves(self):
        """Number of leaves."""
        return sum(child.n_leaves for child in self.children)

    @property
    def height(self):
        """Distance to the deepest descendant."""
        return 1 + max(child.height for child in self.children)

    @property
    def left(self):
        return self.children[0]

    @left.setter
    def left(self, value):
        self.children[0] = value

    @property
    def right(self):
        return self.children[1]

    @right.setter
    def right(self, value):
        self.children[1] = value

    @property
    def mass(self):
        return self.left.mass + self.right.mass

    def next(self, x):
        left, right = self.children
        try:
            value = x[self.feature]
        except (KeyError, IndexError):
            raise ValueError(f"Feature '{self.feature}' is missing in the input")

        if value < self.split_value:
            return left
        return right

    def __repr__(self):
        return str(f"{self.repr_split}@{self.depth}")

    @property
    def depth(self):
        return 1 + self.up.depth if self.up is not None else 0

    def compute_min_feature_value(self):
        return min([inst[self.feature] for inst in self.instances])

    def compute_max_feature_value(self):
        return max([inst[self.feature] for inst in self.instances])

    def remove_instance(self, idx):
        """Remove instance by index."""
        if idx < len(self.instances):
            self.instances.pop(idx)

    def insert_instance(self, x):
        self.instances.append(x)


def H(i):
    """Harmonic number approximation."""
    return math.log(i) + 0.5772156649


def c(n):
    """Average path length of unsuccessful search in BST."""
    return 2 * H(n - 1) - (2 * (n - 1) / n)


class StreamingIsolationTree:
    """A single Streaming Isolation Tree."""

    def __init__(self, X, features, height_limit, rng):
        self.rng = rng
        self.features = features
        self.k = len(X)  # Number of instances to keep in the tree
        self.instances = [x.copy() for x in X]
        self.n = len(X)  # Number of instances seen so far
        self.height_limit = height_limit
        self._root = self._mk_tree(
            self.instances,
            height=self.height_limit,
            rng=self.rng,
            attributes=self.features,
            up=None,
            side=SiBranch.ROOT,
        )

    def _mk_tree(self, X, *, height, rng, attributes=None, up=None, side=SiBranch.ROOT):
        """Build a tree recursively."""
        _attributes = attributes.copy()

        if height == 0 or len(X) == 1:
            return SiLeaf(X, up=up, side=side)

        while len(_attributes) > 0:
            on = rng.choice(_attributes)
            a = float(min([inst[on] for inst in X]))
            b = float(max([inst[on] for inst in X]))
            if a != b:
                break
            _attributes.remove(on)
        else:
            return SiLeaf(X, up=up, side=side)

        at = rng.uniform(a, b)

        # Build the left node
        left = self._mk_tree(
            [inst for inst in X if inst[on] < at],
            height=height - 1,
            rng=rng,
            attributes=attributes,
            up=up,
            side=SiBranch.LEFT,
        )

        # Build the right node
        right = self._mk_tree(
            [inst for inst in X if inst[on] >= at],
            height=height - 1,
            rng=rng,
            attributes=attributes,
            up=up,
            side=SiBranch.RIGHT,
        )

        branch = SiBranch(X, left, right, on, at, up, side, a, b)
        left.up = right.up = branch
        return branch

    def score_instance(self, x):
        """Calculate anomaly score for an instance."""
        score = 0.0

        for node in self._root.walk(x):
            score += 1

        if node.mass > 1:
            score += c(node.mass)

        return score

    def train(self, x):
        """Train the tree with a new instance using reservoir sampling."""
        if self.n < self.k:
            i = self.n
        else:
            i = self.rng.randint(0, self.n)

        if i < self.k:
            x_ = self.instances[i]
            self.instances[i] = x.copy()
            self.remove_instance(x_)
            self.insert_instance(x)

        self.n += 1

        return self

    def insert_instance(self, x):
        """Insert an instance into the tree."""
        # Traverse x to find the feature range not including x and then regrow the tree from there
        for node in self._root.walk(x):
            if isinstance(node, SiBranch):
                node.insert_instance(x)
                if (
                    node.compute_min_feature_value() < node.feature_min
                    or node.compute_max_feature_value() > node.feature_max
                ):
                    break
        else:
            # Leaf node
            node.insert_instance(x)
            if node.depth == self.height_limit:
                # We are at bottom, just insert the new instance
                return

        # Here we either find a node does not contain the new instance with its feature range,
        # or a leaf not being at bottom. We should regrow the subtree
        X = node.instances

        h = self.height_limit - node.depth
        if node.side == SiBranch.ROOT:
            # The parent is the root
            self._root = self._mk_tree(
                X,
                height=h,
                rng=self.rng,
                attributes=self.features,
                up=None,
                side=SiBranch.ROOT,
            )
        elif node.side == SiBranch.LEFT:
            node.up.left = self._mk_tree(
                X,
                height=h,
                rng=self.rng,
                attributes=self.features,
                up=node.up,
                side=SiBranch.LEFT,
            )
        else:  # node.side == SiBranch.RIGHT
            node.up.right = self._mk_tree(
                X,
                height=h,
                rng=self.rng,
                attributes=self.features,
                up=node.up,
                side=SiBranch.RIGHT,
            )

    def remove_instance(self, x):
        """Remove an instance from the tree."""
        # Traverse x to find the feature range change by removing x and then regrow the tree from there
        for node in self._root.walk(x):
            if isinstance(node, SiBranch):
                # Find and remove the instance
                for idx, inst in enumerate(node.instances):
                    if np.array_equal(inst, x):
                        node.remove_instance(idx)
                        break

                if len(node.instances) > 0 and (
                    node.compute_min_feature_value() > node.feature_min
                    or node.compute_max_feature_value() < node.feature_max
                ):
                    break
        else:
            # Leaf node
            for idx, inst in enumerate(node.instances):
                if np.array_equal(inst, x):
                    node.remove_instance(idx)
                    break

            if node.mass > 0:
                return
            else:
                node = node.up

        if node is None or len(node.instances) == 0:
            return

        X = node.instances

        h = self.height_limit - node.depth
        if node.side == SiBranch.ROOT:
            # The parent is the root
            self._root = self._mk_tree(
                X,
                height=h,
                rng=self.rng,
                attributes=self.features,
                up=None,
                side=SiBranch.ROOT,
            )
        elif node.side == SiBranch.LEFT:
            node.up.left = self._mk_tree(
                X,
                height=h,
                rng=self.rng,
                attributes=self.features,
                up=node.up,
                side=SiBranch.LEFT,
            )
        else:  # node.side == SiBranch.RIGHT
            node.up.right = self._mk_tree(
                X,
                height=h,
                rng=self.rng,
                attributes=self.features,
                up=node.up,
                side=SiBranch.RIGHT,
            )


class StreamingIsolationForest(BaseModel):
    """Streaming Isolation Forest for anomaly detection.

    Streaming Isolation Forest constructs an ensemble of isolation trees
    incrementally in a streaming manner to perform anomaly detection. Each tree
    employs reservoir sampling to maintain a fixed-size window of training instances.

    Args:
        window_size (int): The size of the window for each tree (Default=256).
        n_trees (int): The number of trees in the ensemble (Default=100).
        height (int): The maximum height of each tree. If None, it will be set to
            log2(window_size) (Default=None).
        seed (int): Random seed for reproducibility (Default=None).

    References:
        Liu, J.J., Cassales, G.W., Liu, F.T., Pfahringer, B., Bifet, A. (2025).
        Streaming Isolation Forest. In: Wu, X., et al. Advances in Knowledge Discovery
        and Data Mining. PAKDD 2025. Lecture Notes in Computer Science(), vol 15870.
        Springer, Singapore. https://doi.org/10.1007/978-981-96-8170-9_8
    """

    def __init__(self, window_size=256, n_trees=100, height=None, seed=None):
        self.window_size = window_size
        self.n_trees = n_trees
        self.height = height
        self.seed = seed
        self.height_limit = height or math.ceil(math.log2(window_size))
        self._trees = []
        self.instances = []
        self.rng = random.Random(self.seed)
        self.n_features = None

    def fit_partial(self, X, y=None):
        """Fits the model to the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored. Included for API consistency.

        Returns:
            object: Returns self.
        """
        X = np.array(X, dtype=np.float64)

        # Initialize number of features on first instance
        if self.n_features is None:
            self.n_features = len(X)

        if len(self._trees) == 0:
            self.instances.append(X.copy())
            if len(self.instances) == self.window_size:
                features = list(range(self.n_features))
                for _ in range(self.n_trees):
                    t = StreamingIsolationTree(
                        self.instances,
                        features,
                        self.height_limit,
                        self.rng,
                    )
                    self._trees.append(t)
                self.instances = []
        else:
            for t in self._trees:
                t.train(X)

        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score.

        Returns:
            float: The anomalousness score of the input instance. Higher scores
                indicate more anomalous instances.
        """
        X = np.array(X, dtype=np.float64)

        if len(self._trees) == 0:
            return 0.5

        score = 0.0
        for t in self._trees:
            score += t.score_instance(X)

        score /= len(self._trees)
        score /= c(self.window_size)
        score = 2 ** -score

        return score
