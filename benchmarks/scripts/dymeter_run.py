import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import random
import time
import psutil
import json
import argparse
from types import SimpleNamespace
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from utils import get_xy_from_npz, compute_precision_at_k, compute_precision_at_k_windowed

import torch
import torch.nn as nn
import torch.nn.init as init
from METER.compat import patch_hypnettorch
from DyMETER import DyMETER, DynamicThreshold

# upstream METER (which DyMETER extends) expects hypnettorch's MLP.forward to also return the weights
patch_hypnettorch()

import warnings
warnings.filterwarnings("ignore")


def run_dymeter(dataset_name, run_count=None, seed=42, progress_interval=1000, temporal=False,
                output_dir="benchmark_results", verbose=True,
                init_mode="unsupervised", train_rate=0.2, epochs=1000, lr=1e-2, hlr=1e-2,
                emb_rate=2, encoder_layer=10, decoder_layer=10, hyperenc_layer=100, hyperdec_layer=100,
                thres_rate=0.05, mu_e=0.1, edl_filter_rate=0.1, beta_e=0.3, alpha=0.5,
                lam_score=0.6, focal_gamma=2.0, tau=0.95, kappa=0.8, dto_window=64, pivot_decay=0.99,
                update_window=64, update_frac=0.2, update_static_epochs=200, update_hybrid_epochs=500):
    """Run DyMETER (unofficial re-implementation, see src/DyMETER/README.md) on a dataset.

    Training, init modes and offline updating follow meter_run.py (official METER loop). On top:
    focal IEC loss, U_ctr > mu_e routing, Eq. (15) anomaly score and Dynamic Threshold Optimization.
    AUC is computed on the Eq. (15) scores of the scored (non-training) points.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = torch.device("cpu")

    os.makedirs(output_dir, exist_ok=True)
    label = "DyMETER" if init_mode == "unsupervised" else "DyMETER-official"

    print(f"\n===Running {label} on {dataset_name}===")

    try:
        X, y = get_xy_from_npz("../../datasets/" + dataset_name + ".npz")
        if not temporal:  # --temporal keeps the original stream order (known-drift streams)
            X, y = shuffle(X, y, random_state=seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)

        N = int(train_rate * len(X))
        if init_mode == "official":
            train_idx = np.where(y == 0)[0][:N]
        elif init_mode == "unsupervised":
            train_idx = np.arange(N)
        else:
            raise ValueError(f"unknown init_mode {init_mode}")
        test_mask = np.ones(len(X), dtype=bool)
        test_mask[train_idx] = False
        X_test, y_test = X[test_mask], y[test_mask]

        if verbose:
            print(f"Dataset: {X.shape[0]} instances, {X.shape[1]} features, anomaly ratio {y.mean():.4f}")
            print(f"init_mode={init_mode}: {len(train_idx)} training points, {len(X_test)} scored points")

        # METER's hybrid+edl training reads args.mu_e as the pseudo-label filtering rate
        args = SimpleNamespace(mode="hybrid+edl", uncertainty_threshold=mu_e, mu_e=edl_filter_rate, beta_e=beta_e)
        params = {'train_rate': N, 'batch_size': 1, 'lr': lr}
        model = DyMETER(args, X.shape[1], params, alpha, N=N, lr=lr, hlr=hlr, device=device, emb_rate=emb_rate,
                        encoder_layer=encoder_layer, decoder_layer=decoder_layer,
                        hyperenc_layer=hyperenc_layer, hyperdec_layer=hyperdec_layer,
                        lam_score=lam_score, focal_gamma=focal_gamma).to(device)

        def weights_init(m):
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight.data)
        model.apply(weights_init)

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        def train(data, static_epochs, hybrid_epochs):
            torch.set_grad_enabled(True)
            w_enc, w_dec = model.train_autoencoder(data, epochs=static_epochs, mode='static')
            model.train_autoencoder(data, epochs=hybrid_epochs, mode='hybrid+edl',
                                    static_encoder_weight=w_enc, static_decoder_weight=w_dec, thres_rate=thres_rate)
            torch.set_grad_enabled(False)
            return w_enc, w_dec

        # ---- offline training on the historical part ----
        init_data = torch.from_numpy(X[train_idx])
        model.init_data = init_data
        w_enc, w_dec = train(init_data, epochs, 2 * epochs)

        tr = [model.reconstruct(init_data[i:i + 1], w_enc, w_dec, mu_e) for i in range(len(init_data))]
        tr_re = np.array([t[0] for t in tr])
        tr_u = np.array([t[1] for t in tr])
        pivot = float(tr_re.max())                               # r_e initialised to max training error
        mu_t = float(tr_u.max())                                 # max concept uncertainty on training set
        tr_scores = np.array([model.anomaly_score(r, u, pivot) for r, u in zip(tr_re, tr_u)])
        dto = DynamicThreshold(init_threshold=float(tr_scores.max()), mu_t=mu_t, tau=tau, kappa=kappa,
                               window=dto_window)
        uncertainty_avg_threshold = mu_t
        train_time = time.time() - start_time

        # ---- online scoring (Algorithm 2) ----
        X_test_t = torch.from_numpy(X_test)
        all_scores, all_labels, decisions = [], [], []
        scores, times, memory_usages = [], [], []
        uncertainty_list = []
        n_dynamic = 0
        update_time = 0
        max_count = int(round(update_frac * update_window))
        for i in range(len(X_test_t)):
            data = X_test_t[i:i + 1]
            r_e, u_ctr, used_dyn = model.reconstruct(data, w_enc, w_dec, mu_e)
            n_dynamic += used_dyn
            score = model.anomaly_score(r_e, u_ctr, pivot)
            is_anom = dto.update(score, u_ctr)
            if not is_anom:
                pivot = pivot_decay * pivot + (1 - pivot_decay) * r_e

            # offline updating (Eq. 20, as in the official METER loop)
            uncertainty_list.append(u_ctr)
            if i >= update_window:
                recent = uncertainty_list[i - update_window:i]
                if sum(1 for d in recent if d > uncertainty_avg_threshold) > max_count:
                    w_enc, w_dec = train(X_test_t[i - update_window:i], update_static_epochs, update_hybrid_epochs)
                    update_time += 1
                    uncertainty_avg_threshold = alpha * uncertainty_avg_threshold + (1 - alpha) * float(np.mean(uncertainty_list))
                    dto.reset()                                  # re-initialise threshold statistics on drift
                    r_e, u_ctr, _ = model.reconstruct(data, w_enc, w_dec, mu_e)
                    score = model.anomaly_score(r_e, u_ctr, pivot)

            all_scores.append(float(score))
            all_labels.append(float(y_test[i]))
            decisions.append(int(is_anom))

            n = i + 1
            if n % progress_interval == 0:
                yl = np.array(all_labels)
                current_auroc = roc_auc_score(yl, all_scores) if 0 < yl.sum() < len(yl) else 0.5
                scores.append(current_auroc)
                times.append(time.time() - start_time)
                memory_usages.append(abs((psutil.Process().memory_info().rss - start_memory) / 1024 / 1024))
                if verbose:
                    tag = f"(Run {run_count})" if run_count is not None else ""
                    print(f"{label} on {dataset_name} {tag}: {n} instances, AUROC: {current_auroc:.4f}, "
                          f"dynamic: {n_dynamic}, updates: {update_time}, Time: {times[-1]:.2f}s", flush=True)

        precision_k = compute_precision_at_k(all_scores, all_labels)
        precision_k_scores = compute_precision_at_k_windowed(all_scores, all_labels, progress_interval)
        final_auroc = roc_auc_score(all_labels, all_scores)
        total_runtime = time.time() - start_time
        memory_usage = abs((psutil.Process().memory_info().rss - start_memory) / 1024 / 1024)

        results = {
            "dataset": dataset_name,
            "model": label,
            "auc": final_auroc,
            "precision_at_k": precision_k,
            "threshold_decisions": {
                "f1": float(f1_score(all_labels, decisions, zero_division=0)),
                "precision": float(precision_score(all_labels, decisions, zero_division=0)),
                "recall": float(recall_score(all_labels, decisions, zero_division=0)),
                "flagged_rate": float(np.mean(decisions)),
            },
            "total_runtime": total_runtime,
            "total_memory_usage": memory_usage,
            "Instances": len(all_scores),
            "parameters": {
                "init_mode": init_mode, "train_rate": train_rate, "n_train": int(len(train_idx)),
                "epochs": epochs, "lr": lr, "hlr": hlr, "emb_rate": emb_rate, "thres_rate": thres_rate,
                "mu_e": mu_e, "edl_filter_rate": edl_filter_rate, "beta_e": beta_e, "alpha": alpha,
                "lam_score": lam_score, "focal_gamma": focal_gamma, "tau": tau, "kappa": kappa,
                "dto_window": dto_window, "pivot_decay": pivot_decay,
                "update_window": update_window, "update_frac": update_frac,
            },
            "train_time": train_time,
            "dynamic_rate": n_dynamic / max(1, len(all_scores)),
            "model_updates": update_time,
            "implementation": "unofficial re-implementation (src/DyMETER/README.md)",
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
            file_name = f"{label}_{dataset_name}_{run_count}.json"
        else:
            file_name = f"{label}_{dataset_name}.json"

        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        if temporal:  # per-instance scores, needed for post-hoc recovery analysis
            np.savez_compressed(output_file[:-5] + "_scores.npz", scores=np.asarray(all_scores),
                                labels=np.asarray(all_labels), index=np.where(test_mask)[0][:len(all_scores)])

        print(f"\n{label} completed on {dataset_name}")
        print(f"Final AUROC: {final_auroc:.4f} (scored points only), dynamic rate: {results['dynamic_rate']:.3f}, "
              f"updates: {update_time}, threshold F1: {results['threshold_decisions']['f1']:.3f}")
        print(f"Runtime: {total_runtime:.2f}s (training {train_time:.2f}s), Memory: {memory_usage:.2f}MB")
        print(f"Results saved to {output_file}")

        return results

    except Exception as e:
        print(f"Error running {label} on {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run DyMETER (unofficial re-implementation) on a dataset')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--run_count', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--progress_interval', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default="benchmark_results")
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--temporal', action='store_true', help='keep original stream order (no shuffle)')
    parser.add_argument('--init_mode', type=str, default="unsupervised", choices=["unsupervised", "official"])
    parser.add_argument('--train_rate', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=1000)

    args = parser.parse_args()

    run_dymeter(
        dataset_name=args.dataset,
        run_count=args.run_count,
        seed=args.seed,
        progress_interval=args.progress_interval,
        output_dir=args.output_dir,
        verbose=args.verbose,
        temporal=args.temporal,
        init_mode=args.init_mode,
        train_rate=args.train_rate,
        epochs=args.epochs,
    )
