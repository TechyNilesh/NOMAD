import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')   # ARCUS targets tf.keras 2 (upstream: TF 2.2)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import random
import time
import psutil
import json
import argparse
from types import SimpleNamespace
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
# import the benchmark utils before ARCUS, which puts its own utils.py on sys.path
from utils import get_xy_from_npz, compute_precision_at_k, compute_precision_at_k_windowed

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
from ARCUS import ARCUS
from CKA import linear_CKA

import warnings
warnings.filterwarnings("ignore")


def _reduce_models_last(self, x_inp, epochs):
    """Upstream ARCUS._reduce_models_last with the recursive call fixed (upstream passes
    self.model_pool as an extra argument, which raises and aborts the simulation)."""
    latents = [m.get_latent(x_inp).numpy() for m in self.model_pool]
    max_CKA, max_Idx1, max_Idx2 = 0, None, len(latents) - 1
    for idx1 in range(len(latents) - 1):
        CKA = linear_CKA(latents[idx1], latents[max_Idx2])
        if CKA > max_CKA:
            max_CKA, max_Idx1 = CKA, idx1
    if max_Idx1 is not None and max_CKA >= self._similarity_thred:
        self.model_pool[max_Idx2] = self._merge_models(self.model_pool[max_Idx1], self.model_pool[max_Idx2])
        self._train_model(self.model_pool[max_Idx2], x_inp, epochs)
        self.model_pool.remove(self.model_pool[max_Idx1])
        if len(self.model_pool) > 1:
            self._reduce_models_last(x_inp, epochs)


ARCUS._reduce_models_last = _reduce_models_last


def run_arcus(dataset_name, run_count=None, seed=42, progress_interval=1000, temporal=False,
              output_dir="benchmark_results", verbose=True,
              model_type="RAPP", inf_type="ADP", batch_size=512, min_batch_size=32,
              init_epoch=5, intm_epoch=1, layer_num=3, learning_rate=1e-4,
              reliability_thred=0.95, similarity_thred=0.80, pca_variance=0.70):
    """Run ARCUS (official code) on a dataset; re-implements upstream main.py.

    Features are standardised with first-batch statistics and the AE latent size is the number of
    principal components explaining `pca_variance` of the first batch (paper default 70%).
    """

    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n===Running ARCUS on {dataset_name}===")

    try:
        X, y = get_xy_from_npz("../../datasets/" + dataset_name + ".npz")
        if not temporal:  # --temporal keeps the original stream order (known-drift streams)
            X, y = shuffle(X, y, random_state=seed)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)

        first = X[:batch_size]
        mu, sd = first.mean(0), first.std(0)
        sd[sd < 1e-12] = 1.0
        Xs = (X - mu) / sd
        Fs = Xs[:batch_size]
        pca = PCA().fit(Fs)
        hidden_dim = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), pca_variance) + 1)
        hidden_dim = max(1, min(hidden_dim, X.shape[1] - 1)) if X.shape[1] > 1 else 1

        if verbose:
            print(f"Dataset: {X.shape[0]} instances, {X.shape[1]} features, anomaly ratio {y.mean():.4f}")
            print(f"hidden_dim={hidden_dim} (PCA {pca_variance:.0%} on first batch)")

        args = SimpleNamespace(model_type=model_type, inf_type=inf_type, seed=seed, batch_size=batch_size,
                               min_batch_size=min_batch_size, init_epoch=init_epoch, intm_epoch=intm_epoch,
                               hidden_dim=hidden_dim, layer_num=layer_num, learning_rate=learning_rate,
                               reliability_thred=reliability_thred, similarity_thred=similarity_thred,
                               RSRAE_hidden_layer_size=[32, 64, 128], intrinsic_size=hidden_dim,
                               input_dim=X.shape[1], device="cpu")
        # upstream utils.set_seed
        tf.keras.backend.set_floatx('float64')
        tf.random.set_seed(seed)

        model = ARCUS(args)
        n_models = {"created": 0}
        orig_init = model._model_generator.init_model

        def counting_init():
            n_models["created"] += 1
            return orig_init()
        model._model_generator.init_model = counting_init

        loader = tf.data.Dataset.from_tensor_slices((Xs, y.astype(np.float64)))

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        ok, auc_hist, all_scores = model.simulator(loader)
        total_runtime = time.time() - start_time
        if not ok:
            raise RuntimeError("ARCUS simulator reported an error (see message above)")

        all_scores = np.nan_to_num(np.asarray(all_scores, dtype=np.float64))
        all_labels = y[:len(all_scores)].astype(float)

        scores, times, memory_usages = [], [], []
        for n in range(progress_interval, len(all_scores) + 1, progress_interval):
            yl = all_labels[:n]
            scores.append(float(roc_auc_score(yl, all_scores[:n])) if 0 < yl.sum() < n else 0.5)
            times.append(total_runtime * n / len(all_scores))     # batch processing: runtime prorated
            memory_usages.append(abs((psutil.Process().memory_info().rss - start_memory) / 1024 / 1024))

        precision_k = compute_precision_at_k(list(all_scores), list(all_labels))
        precision_k_scores = compute_precision_at_k_windowed(list(all_scores), list(all_labels), progress_interval)
        final_auroc = float(roc_auc_score(all_labels, all_scores))
        memory_usage = abs((psutil.Process().memory_info().rss - start_memory) / 1024 / 1024)

        results = {
            "dataset": dataset_name,
            "model": "ARCUS",
            "auc": final_auroc,
            "auc_batch_mean": float(np.mean(auc_hist)) if auc_hist else None,
            "precision_at_k": precision_k,
            "total_runtime": total_runtime,
            "total_memory_usage": memory_usage,
            "Instances": int(len(all_scores)),
            "parameters": {
                "model_type": model_type, "inf_type": inf_type, "batch_size": batch_size,
                "min_batch_size": min_batch_size, "init_epoch": init_epoch, "intm_epoch": intm_epoch,
                "hidden_dim": hidden_dim, "layer_num": layer_num, "learning_rate": learning_rate,
                "reliability_thred": reliability_thred, "similarity_thred": similarity_thred,
            },
            "models_created": n_models["created"],
            "drift_events": max(0, n_models["created"] - 1),
            "final_pool_size": len(model.model_pool),
            "window_metric": {
                "window_size": progress_interval,
                "auc_scores": scores,
                "precision_at_k_scores": precision_k_scores,
                "runtimes": times,
                "memory_usages": memory_usages,
            }
        }

        if run_count is not None:
            results["run_count"] = run_count
            results["seed"] = seed
            file_name = f"ARCUS_{dataset_name}_{run_count}.json"
        else:
            file_name = f"ARCUS_{dataset_name}.json"

        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        if temporal:  # per-instance scores, needed for post-hoc recovery analysis
            np.savez_compressed(output_file[:-5] + "_scores.npz", scores=np.asarray(all_scores),
                                labels=np.asarray(all_labels), index=np.arange(len(all_scores)))

        print(f"\nARCUS completed on {dataset_name}")
        print(f"Final AUROC: {final_auroc:.4f} (batch-mean AUC {results['auc_batch_mean']})")
        print(f"Drift events: {results['drift_events']}, final pool size: {results['final_pool_size']}")
        print(f"Runtime: {total_runtime:.2f}s, Memory: {memory_usage:.2f}MB")
        print(f"Results saved to {output_file}")

        return results

    except Exception as e:
        print(f"Error running ARCUS on {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ARCUS on a dataset')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--run_count', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--progress_interval', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default="benchmark_results")
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--temporal', action='store_true', help='keep original stream order (no shuffle)')
    parser.add_argument('--model_type', type=str, default="RAPP", choices=["RAPP", "RSRAE", "DAGMM"])
    parser.add_argument('--batch_size', type=int, default=512)

    args = parser.parse_args()

    run_arcus(
        dataset_name=args.dataset,
        run_count=args.run_count,
        seed=args.seed,
        progress_interval=args.progress_interval,
        output_dir=args.output_dir,
        verbose=args.verbose,
        temporal=args.temporal,
        model_type=args.model_type,
        batch_size=args.batch_size,
    )
