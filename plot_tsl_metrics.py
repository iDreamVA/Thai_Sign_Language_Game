"""Plot TSL evaluation artifacts. Run: python plot_tsl_metrics.py MODEL_DIRECTORY."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, cohen_kappa_score,
    confusion_matrix, f1_score, matthews_corrcoef,
    precision_recall_curve, precision_recall_fscore_support, roc_curve,
)


def configure_thai_font(directory):
    for family in ("Tahoma", "Leelawadee UI", "Sarabun", "Noto Sans Thai", "Garuda", "Loma"):
        try:
            font_path = font_manager.findfont(family, fallback_to_default=False)
            break
        except ValueError:
            continue
    else:
        # Colab may need a Thai font. Cache the Google Fonts file beside the plots.
        from urllib.request import urlopen

        directory.mkdir(parents=True, exist_ok=True)
        font_path = directory / "Sarabun-Regular.ttf"
        if not font_path.is_file():
            url = "https://raw.githubusercontent.com/google/fonts/main/ofl/sarabun/Sarabun-Regular.ttf"
            with urlopen(url, timeout=30) as response:
                data = response.read()
            temporary = font_path.with_suffix(".part")
            temporary.write_bytes(data)
            font_manager.FontProperties(fname=str(temporary)).get_name()
            temporary.replace(font_path)
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def save_figure(fig, directory, name, show):
    from tempfile import NamedTemporaryFile

    fig.tight_layout()
    for extension in ("png", "pdf"):
        destination = directory / f"{name}.{extension}"
        with NamedTemporaryFile(prefix=f"{name}_", suffix=f".{extension}", dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            fig.savefig(temporary, format=extension, dpi=180, bbox_inches="tight")
            try:
                temporary.replace(destination)
            except PermissionError:
                print(f"File is open or locked. Saved a new copy to {temporary}")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    if show:
        plt.show()
    plt.close(fig)


def write_table(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decision(probabilities, threshold, null_index):
    return np.where(probabilities.max(axis=1) < threshold, null_index, probabilities.argmax(axis=1))


def validate_predictions(truth, scores, classes):
    truth, scores = np.asarray(truth), np.asarray(scores)
    if truth.ndim != 1 or not np.issubdtype(truth.dtype, np.integer) or not len(truth):
        raise ValueError("Targets must be a nonempty integer vector")
    if scores.shape != (len(truth), classes) or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite probabilities with shape [samples, classes]")
    if truth.min() < 0 or truth.max() >= classes:
        raise ValueError("Target index outside the class mapping")
    if (scores < 0).any() or (scores > 1).any() or not np.allclose(scores.sum(axis=1), 1, atol=1e-4):
        raise ValueError("Expected softmax probabilities summing to one")


def plot_history(history_path, directory, show):
    directory.mkdir(parents=True, exist_ok=True)
    with history_path.open(encoding="utf-8-sig", newline="") as stream:
        history = list(csv.DictReader(stream))
    if not history:
        raise ValueError("history.csv is empty")
    epochs = np.array([float(row["epoch"]) for row in history])
    train_loss_key = "train_eval_loss" if "train_eval_loss" in history[0] else "train_loss"

    def draw_pair(axis, title, columns):
        missing = []
        for label, column in zip(("Training", "Validation"), columns):
            if column in history[0]:
                axis.plot(epochs, [float(row[column]) for row in history],
                          label=label, linewidth=2, marker=".")
            else:
                missing.append(label)
        axis.set(title=title, xlabel="Epoch", ylabel=title)
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend()
        if missing:
            axis.text(0.03, 0.03, f"{' and '.join(missing)} values were not recorded",
                      transform=axis.transAxes, fontsize=9,
                      bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    draw_pair(axes[0], "Loss", (train_loss_key, "val_loss"))
    draw_pair(axes[1], "Accuracy", ("train_accuracy", "val_accuracy"))
    axes[1].set_ylim(0, 1.05)
    if train_loss_key == "train_loss":
        fig.suptitle("Training loss uses the recorded weighted optimization objective", fontsize=10)
    save_figure(fig, directory, "training_validation_loss_accuracy", show)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    specifications = [
        ("Cross-entropy (evaluation mode)", ("train_eval_loss", "val_loss")),
        ("Accuracy", ("train_accuracy", "val_accuracy")),
        ("Macro precision", ("train_macro_precision", "val_macro_precision")),
        ("Macro recall", ("train_macro_recall", "val_macro_recall")),
        ("Macro F1", ("train_macro_f1", "val_macro_f1")),
        ("Weighted F1", ("train_weighted_f1", "val_weighted_f1")),
    ]
    for axis, (title, columns) in zip(axes.flat, specifications):
        draw_pair(axis, title, columns)
    save_figure(fig, directory, "learning_curves", show)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, key, title in zip(axes, ("train_loss", "learning_rate", "seconds"),
                                ("Weighted optimization loss", "Learning rate", "Seconds per epoch")):
        if key in history[0]:
            axis.plot(epochs, [float(row[key]) for row in history], marker=".")
        axis.set(title=title, xlabel="Epoch")
        axis.grid(alpha=0.2)
    save_figure(fig, directory, "training_efficiency", show)


def plot_classification(truth, predicted, classes, threshold, directory, show=False, raw=None):
    directory.mkdir(parents=True, exist_ok=True)
    configure_thai_font(directory.parent / "fonts")
    total = len(classes)
    indexes = np.arange(total)
    codes = [f"C{i:02d}" for i in indexes]
    words = [item["thai"] or item["sign_id"] for item in classes]
    word_counts = {word: words.count(word) for word in set(words)}
    tick_labels = [item["sign_id"] if word_counts[word] > 1 else word
                   for word, item in zip(words, classes)]
    write_table(directory / "class_mapping.csv", [
        {"code": codes[i], "sign_id": item["sign_id"], "thai": item["thai"]}
        for i, item in enumerate(classes)
    ])
    rows = []
    decisions = [("Saved threshold decisions", predicted)] if raw is None else [
        ("Argmax", raw), ("Threshold to null", predicted)]
    for name, pred in decisions:
        row = {"decision": name, "accuracy": accuracy_score(truth, pred)}
        for average in ("macro", "weighted", "micro"):
            p, r, f, _ = precision_recall_fscore_support(
                truth, pred, labels=indexes, average=average, zero_division=0,
            )
            row.update({f"{average}_precision": p, f"{average}_recall": r, f"{average}_f1": f})
        row["mcc"] = matthews_corrcoef(truth, pred)
        row["kappa"] = (cohen_kappa_score(truth, pred, labels=indexes)
                        if len(np.unique(np.concatenate([truth, pred]))) > 1 else None)
        rows.append(row)
    write_table(directory / "summary_metrics.csv", rows)
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "micro_f1"]
    fig, axis = plt.subplots(figsize=(10, 4))
    x = np.arange(len(keys))
    for i, row in enumerate(rows):
        axis.bar(x + (i - 0.5) * 0.36, [row[k] for k in keys], 0.36, label=row["decision"])
    axis.set(xticks=x, xticklabels=keys, ylim=(0, 1.05), title="Test classification metrics")
    axis.tick_params(axis="x", rotation=15)
    axis.legend()
    save_figure(fig, directory, "summary_metrics", show)

    fig, axis = plt.subplots(figsize=(8, 4))
    for i, row in enumerate(rows):
        axis.bar(np.arange(2) + (i - 0.5) * 0.36,
                 [row["mcc"], np.nan if row["kappa"] is None else row["kappa"]],
                 0.36, label=row["decision"])
    axis.axhline(0, color="gray", linewidth=0.8)
    axis.set(xticks=[0, 1], xticklabels=["MCC", "Cohen's kappa"], ylim=(-1.05, 1.05),
             title="Agreement and correlation")
    axis.legend()
    save_figure(fig, directory, "agreement_metrics", show)

    cm = confusion_matrix(truth, predicted, labels=indexes)
    support = cm.sum(axis=1)
    normalized = cm / np.maximum(support[:, None], 1)
    for name, matrix in (("counts", cm), ("row_normalized", normalized)):
        fig, axis = plt.subplots(figsize=(max(10, total * 0.32), max(9, total * 0.32)))
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if name == "row_normalized" else None)
        axis.set(xticks=indexes, yticks=indexes, xticklabels=tick_labels, yticklabels=tick_labels,
                 xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix ({name}, threshold={threshold:.2f})")
        axis.tick_params(axis="x", rotation=90, labelsize=9)
        axis.tick_params(axis="y", labelsize=9)
        if total <= 15:
            for (r, c), value in np.ndenumerate(matrix):
                axis.text(c, r, f"{value:.2f}" if name == "row_normalized" else str(value),
                          ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=axis)
        save_figure(fig, directory, f"confusion_{name}", show)

    p, r, f, support = precision_recall_fscore_support(truth, predicted, labels=indexes, zero_division=0)
    write_table(directory / "per_class.csv", [
        {"code": codes[i], "support": int(support[i]), "precision": p[i], "recall": r[i], "f1": f[i]}
        for i in indexes
    ])
    for start in range(0, total, 20):
        stop = min(start + 20, total)
        fig, axes = plt.subplots(1, 2, figsize=(13, max(4, (stop - start) * 0.35)))
        values = np.arange(stop - start)
        for offset, (name, metric) in enumerate((("Precision", p), ("Recall", r), ("F1", f))):
            axes[0].barh(values + (offset - 1) * 0.23, metric[start:stop], 0.23, label=name)
        axes[0].set(yticks=values, yticklabels=codes[start:stop], xlim=(0, 1.05), title="Per-class scores")
        axes[0].legend()
        axes[1].barh(values, support[start:stop])
        axes[1].set(yticks=values, yticklabels=codes[start:stop], xlabel="Test clips", title="Class support")
        save_figure(fig, directory, f"per_class_{start:02d}", show)

    off_diagonal = cm.copy()
    np.fill_diagonal(off_diagonal, 0)
    pairs = np.argwhere(off_diagonal > 0)
    if len(pairs):
        pairs = sorted(pairs, key=lambda pair: off_diagonal[tuple(pair)], reverse=True)[:20]
        fig, axis = plt.subplots(figsize=(9, 6))
        axis.barh([f"{codes[a]} -> {codes[b]}" for a, b in pairs], [cm[a, b] for a, b in pairs])
        axis.invert_yaxis()
        axis.set(title="Most frequent confusions", xlabel="Clips", ylabel="Actual -> Predicted")
        save_figure(fig, directory, "error_pairs", show)

    return rows


def plot_evaluation(truth, probabilities, classes, threshold, null_index, directory, show=False):
    validate_predictions(truth, probabilities, len(classes))
    directory.mkdir(parents=True, exist_ok=True)
    total = len(classes)
    indexes = np.arange(total)
    codes = [f"C{i:02d}" for i in indexes]
    raw = probabilities.argmax(axis=1)
    predicted = decision(probabilities, threshold, null_index)
    confidence = probabilities.max(axis=1)
    correct = raw == truth
    one_hot = (truth[:, None] == indexes).astype(int)
    rows = plot_classification(truth, predicted, classes, threshold, directory, show, raw)
    support = np.bincount(truth, minlength=total)

    valid_classes = [i for i in indexes if 0 < one_hot[:, i].sum() < len(truth)]
    ranking_rows = []
    for start in range(0, len(valid_classes), 12):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for i in valid_classes[start:start + 12]:
            fp, tp, _ = roc_curve(one_hot[:, i], probabilities[:, i])
            precision, recall, _ = precision_recall_curve(one_hot[:, i], probabilities[:, i])
            roc_auc = auc(fp, tp)
            ap = average_precision_score(one_hot[:, i], probabilities[:, i])
            axes[0].plot(fp, tp, label=f"{codes[i]} AUC={roc_auc:.3f}")
            axes[1].plot(recall, precision, label=f"{codes[i]} AP={ap:.3f}")
            ranking_rows.append({"code": codes[i], "roc_auc_ovr": roc_auc, "average_precision": ap})
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="One-vs-rest ROC")
        axes[1].set(xlabel="Recall", ylabel="Precision", title="One-vs-rest Precision-Recall")
        for axis in axes:
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1.03)
            axis.legend(fontsize=7)
        save_figure(fig, directory, f"roc_pr_classes_{start:02d}", show)
    if ranking_rows:
        write_table(directory / "ranking_metrics.csv", ranking_rows)
    # Micro curves include all model outputs, also for subsets missing some classes.
    fp, tp, _ = roc_curve(one_hot.ravel(), probabilities.ravel())
    precision, recall, _ = precision_recall_curve(one_hot.ravel(), probabilities.ravel())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(fp, tp, label=f"Micro AUC={auc(fp, tp):.3f}")
    if valid_classes:
        grid = np.linspace(0, 1, 1001)
        macro_tp = []
        for i in valid_classes:
            fpr, tpr, _ = roc_curve(one_hot[:, i], probabilities[:, i])
            macro_tp.append(np.interp(grid, fpr, tpr))
        axes[0].plot(grid, np.mean(macro_tp, axis=0), label="Macro ROC (defined classes)")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Aggregate ROC")
    axes[1].plot(recall, precision, label=f"Micro AP={average_precision_score(one_hot.ravel(), probabilities.ravel()):.3f}")
    axes[1].axhline(1 / total, color="gray", linestyle="--", label="Micro prevalence")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Aggregate Precision-Recall")
    for axis in axes:
        axis.legend()
    save_figure(fig, directory, "roc_pr_aggregate", show)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(0, 1, 11)
    for name, mask in (("Correct", correct), ("Incorrect", ~correct)):
        if mask.any():
            axes[0].hist(confidence[mask], bins=bins, alpha=0.6, label=name)
    axes[0].set(xlabel="Top softmax score", ylabel="Clips", title="Argmax confidence distribution")
    axes[0].legend()
    bin_id = np.minimum((confidence * 10).astype(int), 9)
    bin_rows = []
    for i in range(10):
        mask = bin_id == i
        if mask.any():
            bin_rows.append({"bin": i, "count": int(mask.sum()),
                             "confidence": float(confidence[mask].mean()),
                             "accuracy": float(correct[mask].mean())})
    ece = sum(abs(row["accuracy"] - row["confidence"]) * row["count"] / len(truth) for row in bin_rows)
    axes[1].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    axes[1].scatter([row["confidence"] for row in bin_rows], [row["accuracy"] for row in bin_rows],
                    s=[20 + 200 * row["count"] / len(truth) for row in bin_rows])
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean top score", ylabel="Observed accuracy",
                title=f"Reliability diagram (10 equal bins), ECE={ece:.3f}")
    save_figure(fig, directory, "confidence_calibration", show)
    write_table(directory / "calibration_bins.csv", bin_rows)
    nll = -np.log(np.clip(probabilities[np.arange(len(truth)), truth], 1e-12, 1)).mean()
    probability_metrics = {"nll": float(nll), "multiclass_brier_sum": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
                           "top_label_ece_10_bins": float(ece), "absent_test_classes": int((support == 0).sum())}
    (directory / "probability_metrics.json").write_text(json.dumps(probability_metrics, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, key, title in zip(axes, ("nll", "multiclass_brier_sum", "top_label_ece_10_bins"),
                                ("Negative log-likelihood", "Multiclass Brier (sum)", "Top-label ECE")):
        value = probability_metrics[key]
        axis.bar([title], [value])
        axis.set(title="Lower is better", ylim=(0, max(value * 1.25, 0.05)))
        axis.text(0, value, f"{value:.4f}", ha="center", va="bottom")
    save_figure(fig, directory, "probability_metrics", show)

    thresholds = np.linspace(0, 1, 51)
    coverage, accuracy = [], []
    for value in thresholds:
        retained = confidence >= value
        coverage.append(retained.mean())
        accuracy.append(correct[retained].mean() if retained.any() else np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(coverage, accuracy, marker=".")
    axes[0].set(xlabel="Retained fraction (includes predicted null)", ylabel="Accuracy on retained clips",
                title="Coverage vs accuracy (test diagnostic)", ylim=(0, 1.05))
    order = np.argsort(-probabilities, axis=1, kind="stable")
    ks = sorted({1, min(3, total), min(5, total), min(10, total)})
    topk = [(order[:, :k] == truth[:, None]).any(axis=1).mean() for k in ks]
    axes[1].bar([str(k) for k in ks], topk)
    axes[1].set(xlabel="k", ylabel="Accuracy", title="Top-k accuracy (raw scores)", ylim=(0, 1.05))
    save_figure(fig, directory, "coverage_topk", show)
    return rows


def generate_report(model_dir, show=False):
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    directory = model_dir / "plots"
    directory.mkdir(exist_ok=True)
    if (model_dir / "history.csv").is_file():
        plot_history(model_dir / "history.csv", directory, show)
    classes = metadata["classes"]
    threshold, null_index = metadata["confidence_threshold"], metadata["null_index"]
    prediction_file = model_dir / "evaluation_predictions.npz"
    if not prediction_file.is_file():
        found = False
        for path in sorted(model_dir.glob("*_confusion.csv")):
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                header, records = next(reader), list(reader)
            labels = [item["sign_id"] for item in classes]
            if header[1:] != labels or [row[0] for row in records] != labels:
                raise ValueError(f"Class order mismatch in {path}")
            matrix = np.array([[int(value) for value in row[1:]] for row in records])
            if matrix.shape != (len(classes), len(classes)) or (matrix < 0).any() or not matrix.sum():
                raise ValueError(f"Invalid confusion matrix in {path}")
            actual, predicted = np.indices(matrix.shape)
            # Recover aggregate decision counts without inventing softmax scores.
            truth = np.repeat(actual.ravel(), matrix.ravel())
            pred = np.repeat(predicted.ravel(), matrix.ravel())
            plot_classification(truth, pred, classes, threshold,
                                directory / path.stem.removesuffix("_confusion"), show)
            found = True
        if not found:
            raise FileNotFoundError("Need evaluation_predictions.npz or *_confusion.csv")
        print("Existing bundle plotted from saved threshold decisions and recorded history.")
        print("ROC/PR, confidence and top-k plots require per-clip softmax scores from the updated evaluation cell.")
        print(f"Saved PNG, PDF, CSV reports to {directory}")
        return directory
    with np.load(prediction_file, allow_pickle=False) as saved:
        for subset in np.unique(saved["test_subset"]):
            mask = saved["test_subset"] == subset
            plot_evaluation(saved["test_truth"][mask], saved["test_scores"][mask], classes,
                            threshold, null_index, directory / str(subset), show)
        truth, scores = saved["val_truth"], saved["val_scores"]
        validate_predictions(truth, scores, len(classes))
        thresholds = np.linspace(0, 1, 101)
        fig, axis = plt.subplots(figsize=(9, 4))
        for name, metric in (("Accuracy", accuracy_score), ("Macro F1", None)):
            values = []
            for value in thresholds:
                pred = decision(scores, value, null_index)
                values.append(metric(truth, pred) if metric else f1_score(
                    truth, pred, labels=np.arange(len(classes)), average="macro", zero_division=0))
            axis.plot(thresholds, values, label=name)
        axis.axvline(threshold, linestyle="--", color="black", label=f"Saved threshold={threshold:.2f}")
        axis.set(xlabel="Confidence threshold mapped to null", ylabel="Score", title="Validation threshold diagnostic")
        axis.legend()
        save_figure(fig, directory, "validation_threshold", show)
    benchmark_file = model_dir / "inference_benchmark.json"
    if benchmark_file.is_file():
        benchmark = json.loads(benchmark_file.read_text(encoding="utf-8"))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(benchmark["latency_ms"], bins=20)
        axes[0].set(xlabel="ms per batch-1 forward pass", ylabel="Runs", title=f"Inference latency on {benchmark['device']}")
        axes[1].bar(["p50", "p95", "p99"], np.percentile(benchmark["latency_ms"], [50, 95, 99]))
        axes[1].set(ylabel="Milliseconds", title="Model latency percentiles")
        save_figure(fig, directory, "inference_latency", show)
    print(f"Saved PNG, PDF, CSV reports to {directory}")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    generate_report(args.model_dir, show=args.show)
