import numpy as np

def compute_precision_at_k(all_scores, all_labels, k_values=None):
    """Compute Precision@k: fraction of true anomalies in the top-k scored instances."""
    scores = np.array(all_scores)
    labels = np.array(all_labels)
    n = len(scores)
    n_anomalies = int(labels.sum())

    if k_values is None:
        # Use absolute k values and percentage-based ones
        k_values = sorted(set([
            100, 200, 500,
            max(1, int(0.01 * n)),  # top 1%
            max(1, int(0.05 * n)),  # top 5%
            max(1, int(0.10 * n)),  # top 10%
        ]))

    top_k_indices = np.argsort(scores)[::-1]
    result = {}
    for k in k_values:
        k = min(k, n)
        top_k_labels = labels[top_k_indices[:k]]
        result[f"precision@{k}"] = float(top_k_labels.sum() / k)

    result["n_anomalies"] = n_anomalies
    result["n_total"] = n
    return result


def compute_precision_at_k_windowed(all_scores, all_labels, progress_interval=1000):
    """Prequential Precision@(top-1%): at each window boundary, compute
    Precision@(top-1% of all instances seen so far) over the cumulative
    score history.

    This mirrors the prequential AUC computation, producing smooth curves
    that track how the model's ranking quality evolves over the stream.
    """
    scores = np.asarray(all_scores)
    labels = np.asarray(all_labels)
    n = len(scores)

    window_scores = []
    for end in range(progress_interval, n + progress_interval, progress_interval):
        end = min(end, n)
        cum_scores = scores[:end]
        cum_labels = labels[:end]
        cum_n = len(cum_scores)
        cum_anom = int(cum_labels.sum())
        k = max(1, int(0.01 * cum_n))
        k = min(k, cum_n)
        if cum_anom == 0 or cum_n < 2:
            window_scores.append(float('nan'))
            continue
        top_idx = np.argsort(cum_scores)[::-1][:k]
        pk = float(cum_labels[top_idx].sum() / k)
        window_scores.append(pk)

    return window_scores


def get_xy_from_npz(file_path, x_key='X', y_key='y'):
    npz_data = np.load(file_path, allow_pickle=True)
    available_keys = npz_data.files
    if x_key not in available_keys:
        raise KeyError(f"X key '{x_key}' not found in NPZ file. Available keys: {available_keys}")
    if y_key not in available_keys:
        raise KeyError(f"y key '{y_key}' not found in NPZ file. Available keys: {available_keys}")
    X = npz_data[x_key]
    y = npz_data[y_key]
    return X, y