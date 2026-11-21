import numpy as np
import random
import math
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from scipy.stats import truncnorm
from scipy.stats import norm

from pysad_rust import (
    HalfSpaceTrees,
    IForestASD,
    RobustRandomCutForest,
    LODA,
    OnlineIsolationForest,
)


# ──────────────────────────  helpers  ──────────────────────────
class PostProcessor:
    """
    Tracks min/max/mean/variance and returns a score normalised to [0,1].
    """

    def __init__(self, max_history: int = 1000):
        self._min = float("inf")
        self._max = float("-inf")
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0
        self.normalized_scores = []  # Store normalized scores for correlation
        self.max_history = max_history

    def process(self, score: float) -> float:
        if score < self._min:
            self._min = score
        if score > self._max:
            self._max = score

        self._count += 1
        self._sum += score
        self._sum_sq += score ** 2

        span = self._max - self._min
        normalized = 0.0 if span == 0 else (score - self._min) / span
        
        # Store normalized score
        self.normalized_scores.append(normalized)
        if len(self.normalized_scores) > self.max_history:
            self.normalized_scores.pop(0)
            
        return normalized

    # Convenience metrics (optional)
    def mean(self):
        return 0.0 if self._count == 0 else self._sum / self._count

    def variance(self):
        if self._count < 2:
            return 0.0
        mean = self.mean()
        return (self._sum_sq / self._count) - mean**2


class SimpleAnomalyRewardCalculator:
    """
    Simple reward calculator that uses anomaly scores directly.
    Higher anomaly scores indicate better anomaly detection performance.
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.recent_scores = []  # List of lists, one per model
        
    def update_scores(self, normalized_scores: list):
        """
        Update normalized scores for all models.
        normalized_scores: list of normalized scores from each model at current time step
        """
        # Initialize if first call
        if not self.recent_scores:
            self.recent_scores = [[] for _ in range(len(normalized_scores))]
            
        # Add scores for each model
        for i, score in enumerate(normalized_scores):
            self.recent_scores[i].append(score)
            # Keep only recent scores
            if len(self.recent_scores[i]) > self.window_size:
                self.recent_scores[i].pop(0)
                
    def calculate_rewards(self) -> np.ndarray:
        """
        Calculate rewards based on recent anomaly scores.
        Returns rewards for each model (higher is better for anomaly detection).
        """
        if not self.recent_scores or len(self.recent_scores[0]) < 10:
            return np.zeros(len(self.recent_scores))
            
        rewards = []
        for model_scores in self.recent_scores:
            if len(model_scores) < 10:
                rewards.append(0.0)
                continue
                
            # Use mean of recent scores as reward
            # Higher anomaly scores are better for anomaly detection
            mean_score = np.mean(model_scores[-min(100, len(model_scores)):])
            rewards.append(mean_score)
                
        return np.array(rewards)


class MCTSNode:
    """
    A node in the Monte Carlo Tree Search tree.
    Each node represents a model selection state.
    """
    
    def __init__(self, model_idx: int = None, parent=None):
        self.model_idx = model_idx  # Index of the model this node represents
        self.parent = parent
        self.children = {}  # Dictionary mapping model_idx to child nodes
        self.visits = 0
        self.total_reward = 0.0
        self.rewards = []  # Store individual rewards for statistics
        
    def is_leaf(self):
        return len(self.children) == 0
    
    def add_child(self, model_idx: int):
        """Add a child node for the given model index."""
        if model_idx not in self.children:
            self.children[model_idx] = MCTSNode(model_idx, parent=self)
        return self.children[model_idx]
    
    def update(self, reward: float):
        """Update node statistics with a new reward."""
        self.visits += 1
        self.total_reward += reward
        self.rewards.append(reward)
        
        # Keep only recent rewards for better adaptation
        if len(self.rewards) > 100:
            self.rewards.pop(0)
    
    def ucb_score(self, c: float = 1.41):
        """Calculate UCB1 score for node selection."""
        if self.visits == 0:
            return float('inf')
        
        avg_reward = self.total_reward / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return avg_reward + exploration
    
    def best_child(self, c: float = 1.41):
        """Select the best child using UCB1."""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda child: child.ucb_score(c))
    
    def most_visited_child(self):
        """Select the child with the most visits."""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda child: child.visits)


class MCTS:
    """
    Monte Carlo Tree Search for model selection.
    """
    
    def __init__(self, n_models: int, c: float = 1.41, max_simulations: int = 100):
        self.n_models = n_models
        self.c = c  # UCB exploration parameter
        self.max_simulations = max_simulations
        self.root = MCTSNode()
        self.model_rewards = np.zeros(n_models)  # Track recent model performance
        
    def select(self, node: MCTSNode):
        """Selection phase: traverse from root to a leaf using UCB1."""
        while not node.is_leaf():
            node = node.best_child(self.c)
        return node
    
    def expand(self, node: MCTSNode):
        """Expansion phase: add all possible children (models) to the leaf."""
        for model_idx in range(self.n_models):
            if model_idx not in node.children:
                node.add_child(model_idx)
        return node
    
    def simulate(self, node: MCTSNode):
        """Simulation phase: random rollout from current node."""
        # For anomaly detection, we use the current model rewards as simulation results
        # In a real implementation, this could involve running the model for several steps
        if node.model_idx is not None:
            return self.model_rewards[node.model_idx]
        else:
            # If root node, return average reward
            return np.mean(self.model_rewards)
    
    def backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagation phase: update all nodes on the path with the reward."""
        while node is not None:
            node.update(reward)
            node = node.parent
    
    def search(self):
        """Run MCTS to find the best model."""
        for _ in range(self.max_simulations):
            # Selection
            leaf = self.select(self.root)
            
            # Expansion
            if leaf.visits > 0:  # Only expand if leaf has been visited
                self.expand(leaf)
                if leaf.children:
                    leaf = random.choice(list(leaf.children.values()))
            
            # Simulation
            reward = self.simulate(leaf)
            
            # Backpropagation
            self.backpropagate(leaf, reward)
    
    def best_action(self):
        """Return the best model index based on MCTS results."""
        if not self.root.children:
            return random.randint(0, self.n_models - 1)
        
        # Choose the most visited child (exploitation)
        best_child = self.root.most_visited_child()
        return best_child.model_idx if best_child else random.randint(0, self.n_models - 1)
    
    def update_model_rewards(self, rewards: np.ndarray):
        """Update the current model rewards for simulation."""
        self.model_rewards = rewards.copy()


# ──────────────────────────  main class  ──────────────────────────
class AutoSAD_MCTS(ABC):
    def __init__(
        self,
        n_models: int = 5,
        random_state: int = 52,
        verbose: bool = False,
        enable_evolution: bool = True,
        enable_diversity: bool = True,
        evolution_interval: int = 1000,
        reward_strategy: str = "simple",  # "simple" for now
        diversity_probability: float = 0.3,
        mcts_c: float = 1.41,
        mcts_simulations: int = 50,
        **kwargs,
    ):
        self.n_models = n_models
        self.random_state = random_state
        self.verbose = verbose
        self.enable_evolution = enable_evolution
        self.enable_diversity = enable_diversity
        self.evolution_interval = evolution_interval
        self.reward_strategy = reward_strategy
        self.diversity_probability = diversity_probability
        self.mcts_c = mcts_c
        self.mcts_simulations = mcts_simulations

        random.seed(random_state)
        np.random.seed(random_state)

        self.feature_mins = kwargs.get("feature_mins")
        self.feature_maxes = kwargs.get("feature_maxes")

        self.models = []
        self.model_params = []
        self.scores = []
        self.step = 0
        self.exploration_std = 1.0

        self._init_pool()
        self.mcts = MCTS(self.n_models, c=self.mcts_c, max_simulations=self.mcts_simulations)
        
        # Initialize reward calculator
        if self.reward_strategy == "simple":
            self.reward_calculator = SimpleAnomalyRewardCalculator()
        else:
            raise ValueError(f"Unknown reward strategy: {self.reward_strategy}")

    # ───────────────────── hyperparameter grids ─────────────────────
    def _get_hyperparameter_options(self):
        return {
            "HalfSpaceTrees": {
                "num_trees": [8, 16, 32, 64, 128],
                "max_depth": [5, 8, 10, 12, 15],
                "window_size": [50, 100, 150, 200, 250],
            },
            "IForestASD": {
                "window_size": [512, 1024, 2048, 4096],
                "n_estimators": [16, 32, 64, 128],
                "max_samples": [64, 128, 256, 512],
                "contamination": [0.05, 0.1, 0.15, 0.2],
                "anomaly_rate_threshold": [0.1, 0.2, 0.3, 0.4],
            },
            "RobustRandomCutForest": {
                "num_trees": [8, 16, 32, 64, 128],
                "tree_size": [64, 128, 256, 512],
                "shingle_size": [1, 2, 3],
            },
            "LODA": {
                "num_bins": [50, 100, 150, 200],
                "num_random_cuts": [5, 10, 25, 50],
            },
            "OnlineIsolationForest": {
                "branching_factor": [2, 3, 4],
                "split": ["axisparallel", "hyperplane"],
                "num_trees": [8, 16, 32, 64, 128],
                "max_leaf_samples": [16, 32, 64],
                "window_size": [512, 1024, 2048, 4096],
                "growth_criterion": ["adaptive", "depth", "size"],
                "subsample": [0.1, 0.25, 0.5, 0.75, 1.0],
            },
        }

    # ───────────────────── weighted HP sampler ─────────────────────
    def _adaptive_hp_sample(self, current, candidates, sigma=1.0):
        """
        Adaptive hyperparameter optimization sampling function.
        
        Dynamically adjusts exploration based on sigma parameter:
        - For numeric parameters: Uses truncated normal distribution centered on current value
        - For categorical parameters: Weights probabilities to favor current value
    
        Args:
            current: Current hyperparameter value
            candidates: List of valid hyperparameter values  
            sigma: Exploration strength (higher = more exploration, lower = more exploitation)
            
        Returns:
            Sampled hyperparameter value with adaptive exploration/exploitation balance
        """
        if isinstance(current, (int, float)):
            lo, hi = min(candidates), max(candidates)
            std = np.std(candidates) * sigma
            a, b = (lo - current) / std, (hi - current) / std
            try:
                new = truncnorm.rvs(a, b, loc=current, scale=std)
                new = min(candidates, key=lambda x: abs(x - new))
                return int(round(new)) if isinstance(current, int) else new
            except ValueError:
                return np.random.choice(candidates)
        # categorical
        p_keep = max(0.5, 1 - 0.5 * sigma)
        probs = [p_keep if v == current else (1 - p_keep) / (len(candidates) - 1)
                 for v in candidates]
        return np.random.choice(candidates, p=probs)

    # ───────────────────── pool initialisation ─────────────────────
    def random_model_with_params(self):
        types = [HalfSpaceTrees, IForestASD, RobustRandomCutForest, LODA, OnlineIsolationForest]
        cls = np.random.choice(types)
        opts = self._get_hyperparameter_options()[cls.__name__]
        params = {"random_state": self.random_state}
        for p, vals in opts.items():
            if p not in ("feature_mins", "feature_maxes"):
                params[p] = np.random.choice(vals)
        if cls is HalfSpaceTrees:
            params["feature_mins"] = self.feature_mins
            params["feature_maxes"] = self.feature_maxes
        return cls(**params), params

    def _init_pool(self):
        while len(self.models) < self.n_models:
            m, p = self.random_model_with_params()
            self.models.append(m)
            self.model_params.append(p)
            self.scores.append(PostProcessor())

    # ───────────────────── mutation ─────────────────────
    def mutate(self, model, params):
        name = type(model).__name__
        opts = self._get_hyperparameter_options()[name]
        new_params = params.copy()
        new_params["random_state"] = self.random_state
        for p, vals in opts.items():
            if p in params and p not in ("feature_mins", "feature_maxes"):
                new_params[p] = self._adaptive_hp_sample(params[p], vals, self.exploration_std)
        if name == "HalfSpaceTrees":
            new_params["feature_mins"] = self.feature_mins
            new_params["feature_maxes"] = self.feature_maxes
            return HalfSpaceTrees(**new_params), new_params
        elif name == "IForestASD":
            return IForestASD(**new_params), new_params
        elif name == "RobustRandomCutForest":
            return RobustRandomCutForest(**new_params), new_params
        elif name == "LODA":
            return LODA(**new_params), new_params
        elif name == "OnlineIsolationForest":
            return OnlineIsolationForest(**new_params), new_params
        return model, params

    # ───────────────────── online step ─────────────────────
    def fit_score_partial(self, X):
        # Get normalized scores from all models
        normalized_scores = []
        
        for i, model in enumerate(self.models):
            raw_score = model.fit_score_partial(X)
            normalized_score = self.scores[i].process(raw_score)
            normalized_scores.append(normalized_score)
        
        # Update reward calculator with normalized scores
        self.reward_calculator.update_scores(normalized_scores)
        
        # Calculate rewards based on anomaly scores
        rewards = self.reward_calculator.calculate_rewards()
        
        # Update MCTS with current model rewards
        self.mcts.update_model_rewards(rewards)
        
        # Run MCTS to select best model
        self.mcts.search()
        selected_model = self.mcts.best_action()
        
        self.step += 1

        # evolution every evolution_interval points (if enabled)
        if self.enable_evolution and self.step % self.evolution_interval == 0:
            self._evolve()

        return normalized_scores[selected_model]

    # ───────────────────── evolution ─────────────────────
    def _evolve(self):
        if self.verbose:
            print(f"EVOLVING at step {self.step}...")
        
        # update exploration-decay
        avg = np.mean([s.mean() for s in self.scores])
        self.exploration_std = np.clip(self.exploration_std * 2 ** (-avg), 0.1, 2.0)

        # Get current model performance from MCTS
        model_performance = []
        for i in range(self.n_models):
            if i in self.mcts.root.children:
                child = self.mcts.root.children[i]
                if child.visits > 0:
                    model_performance.append((i, child.total_reward / child.visits))
                else:
                    model_performance.append((i, 0.0))
            else:
                model_performance.append((i, 0.0))
        
        # Sort by performance (higher is better)
        model_performance.sort(key=lambda x: x[1], reverse=True)
        
        # best model is safe
        best_idx = model_performance[0][0]
        
        # worst model is candidate for replacement
        worst_idx = model_performance[-1][0]
        
        # with some probability, mutate the worst model
        if np.random.rand() < 0.7:  # probability of evolution
            
            # create a new model by mutating the best one
            new_model, new_params = self.mutate(
                self.models[best_idx], self.model_params[best_idx]
            )
            
            # replace worst model
            self.models[worst_idx] = new_model
            self.model_params[worst_idx] = new_params
            self.scores[worst_idx] = PostProcessor()
            
            # Reset MCTS tree to account for model change
            self.mcts = MCTS(self.n_models, c=self.mcts_c, max_simulations=self.mcts_simulations)
            
            if self.verbose:
                print(f"  MUTATED: {type(self.models[best_idx]).__name__} -> {type(new_model).__name__}")
                print(f"  REPLACED model {worst_idx}")

        # diversity mechanism: with some probability, replace a random model
        if self.enable_diversity and np.random.rand() < self.diversity_probability:
            
            # choose a random model to replace (not the best one)
            rand_idx = np.random.choice([i for i in range(self.n_models) if i != best_idx])
            
            # create a new random model
            new_model, new_params = self.random_model_with_params()
            
            # replace random model
            self.models[rand_idx] = new_model
            self.model_params[rand_idx] = new_params
            self.scores[rand_idx] = PostProcessor()
            
            # Reset MCTS tree to account for model change
            self.mcts = MCTS(self.n_models, c=self.mcts_c, max_simulations=self.mcts_simulations)
            
            if self.verbose:
                print(f"  DIVERSIFIED: replaced model {rand_idx} with {type(new_model).__name__}")

        if self.verbose:
            print("pool types:", Counter([type(m).__name__ for m in self.models]))
            print(f"exploration σ = {self.exploration_std:.3f}")

            # print best model and params
            print(f"best model: {type(self.models[best_idx]).__name__}")
            print(f"params: {self.model_params[best_idx]}")
            print(f"score: {self.scores[best_idx].mean():.3f}")

            rewards = self.reward_calculator.calculate_rewards()
            if len(rewards) > 0:
                print(f"reward: {rewards[best_idx]:.3f}")
                
            # Print MCTS statistics
            if best_idx in self.mcts.root.children:
                best_child = self.mcts.root.children[best_idx]
                print(f"MCTS visits: {best_child.visits}")
                print(f"MCTS avg reward: {best_child.total_reward / max(1, best_child.visits):.3f}")
