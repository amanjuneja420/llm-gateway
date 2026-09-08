"""
Train a routing classifier on the manually-judged eval_set.csv.
====================================================================
Run manually, once eval_set.csv's which_is_better column has been filled
in by hand (see generate_eval_set.py - that script only generates the
prompt/response pairs, judging them is separate, manual work):

    venv\\Scripts\\python train_classifier.py

What this does, in order:

1. Load eval_set.csv, drop any "neither" row (no usable preference
   signal - neither response was good, which doesn't tell us anything
   about which MODEL should have been picked).
2. Fold "tie" rows into the "1.5b" label. This is a deliberate choice,
   not a coding shortcut: when both models answer equally well, the
   heuristic's whole point is picking the minimum sufficient model, not
   winning a quality contest - so a tie should route to the cheaper/
   faster model, not be discarded or treated as ambiguous.
3. Embed every remaining prompt with the exact same embedding pipeline
   cache.py already uses (SemanticCache.embed(), which wraps
   sentence-transformers/all-MiniLM-L6-v2 with normalize_embeddings=True)
   - reusing this instead of standing up a second embedding path means
   whatever gets saved here is guaranteed compatible with what main.py
   will feed it at inference time (see the "trained" router mode there).
4. Train a LogisticRegression (default L2 regularization - deliberately
   NOT disabled, since ~49 examples with 384-dim embeddings is a real
   small-n-large-p regime where an unregularized model would overfit
   badly) and evaluate with stratified k-fold cross-validation rather
   than a single train/test split, which would be far too noisy to trust
   at this sample size.
5. Compute what the EXISTING heuristic router (router.py) would have
   predicted for the same rows, and report its accuracy against the same
   ground truth - side by side with the classifier's cross-validated
   accuracy, honestly, whichever way it comes out.
6. Save the final classifier (trained on all available rows, not just
   one fold) to router_classifier.pkl. This is a small artifact this
   project creates and trusts itself, unlike the semantic cache's
   deliberately-not-pickle persistence (see cache.py) - the concern there
   was untrusted data on disk being able to execute code on load; this
   pickle only ever holds a model this script just trained.
"""

import csv
import pickle
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from cache import SemanticCache
from router import route

EVAL_SET_PATH = "eval_set.csv"
CLASSIFIER_PATH = "router_classifier.pkl"

# k=5 is the default; reduced automatically below if the smaller class
# doesn't have at least this many examples (StratifiedKFold requires every
# fold to contain at least one example of each class).
DEFAULT_K_FOLDS = 5


def load_labeled_prompts(path: str = EVAL_SET_PATH) -> tuple[list[str], list[str]]:
    prompts: list[str] = []
    labels: list[str] = []
    excluded_neither = 0
    folded_ties = 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            judgment = row["which_is_better"].strip().lower()
            if judgment == "neither":
                excluded_neither += 1
                continue
            if judgment == "tie":
                label = "1.5b"  # fold ties to the cheaper/faster model - see module docstring
                folded_ties += 1
            elif judgment in ("1.5b", "3b"):
                label = judgment
            else:
                raise ValueError(
                    f"Unexpected which_is_better value {judgment!r} for prompt {row['prompt']!r}"
                )
            prompts.append(row["prompt"])
            labels.append(label)

    print(f"Loaded {len(prompts) + excluded_neither} judged rows from {path}")
    print(f"  excluded (which_is_better='neither'): {excluded_neither}")
    print(f"  folded 'tie' -> '1.5b': {folded_ties}")
    print(f"  remaining for training: {len(prompts)}  (label counts: {dict(Counter(labels))})")
    return prompts, labels


def embed_prompts(prompts: list[str]) -> np.ndarray:
    # Reuses cache.py's exact embedding pipeline (same model, same
    # normalize_embeddings=True) rather than standing up a second one -
    # see module docstring for why that matters for the "trained" router
    # mode in main.py.
    embedder = SemanticCache()
    return np.vstack([embedder.embed(p) for p in prompts])


def heuristic_predictions(prompts: list[str]) -> list[str]:
    model_to_label = {"qwen2.5:1.5b": "1.5b", "qwen2.5:3b": "3b"}
    return [model_to_label[route(p)] for p in prompts]


def main() -> None:
    prompts, labels = load_labeled_prompts()
    label_counts = Counter(labels)
    minority_count = min(label_counts.values())
    k_folds = min(DEFAULT_K_FOLDS, minority_count)
    if k_folds < DEFAULT_K_FOLDS:
        print(
            f"\nSmaller class has only {minority_count} examples - "
            f"reducing k from {DEFAULT_K_FOLDS} to {k_folds} so every fold "
            f"can contain at least one example of it."
        )

    print("\nEmbedding prompts (sentence-transformers/all-MiniLM-L6-v2, via cache.py)...")
    X = embed_prompts(prompts)
    y = np.array(labels)

    print(f"\n{'=' * 64}")
    print("CROSS-VALIDATED CLASSIFIER EVALUATION".center(64))
    print("=" * 64)

    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    model = LogisticRegression()  # default L2 regularization, not disabled - see docstring

    fold_accuracies = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        model.fit(X[train_idx], y[train_idx])
        fold_pred = model.predict(X[test_idx])
        fold_acc = accuracy_score(y[test_idx], fold_pred)
        fold_accuracies.append(fold_acc)
        print(f"  Fold {fold_idx}/{k_folds}: accuracy={fold_acc:.3f}  (n_test={len(test_idx)})")

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    print(f"\n  Mean CV accuracy: {mean_acc:.3f}  (std: {std_acc:.3f})")
    print(
        "  Spread across folds matters at this sample size - a single "
        "number here would hide how much this estimate could shift with "
        "a different split."
    )

    # Aggregated confusion matrix: cross_val_predict gives one out-of-fold
    # prediction per row (each row predicted by a model that never saw it
    # during training), so this reflects genuine held-out performance
    # across the whole dataset, not just one fold.
    y_pred_oof = cross_val_predict(LogisticRegression(), X, y, cv=cv)
    labels_order = ["1.5b", "3b"]
    cm = confusion_matrix(y, y_pred_oof, labels=labels_order)
    print("\n  Confusion matrix (rows=actual, cols=predicted), aggregated across folds:")
    print(f"  {'':>10}{'pred 1.5b':>12}{'pred 3b':>10}")
    for i, row_label in enumerate(labels_order):
        print(f"  {'actual ' + row_label:>10}{cm[i][0]:>12}{cm[i][1]:>10}")

    # Explicit check for a degenerate majority-class predictor - easy to
    # miss by eyeballing the confusion matrix above, but it changes what
    # the accuracy number actually means. At 49 examples with a 32/17
    # class split, a model that always predicts the majority class scores
    # majority_baseline_acc without learning anything from the embeddings
    # at all.
    n_predicted_3b = int((y_pred_oof == "3b").sum())
    majority_baseline_acc = label_counts["1.5b"] / len(y)
    print(f"\n  Predicted '3b' for {n_predicted_3b}/{len(y)} rows across all folds.")
    print(f"  Majority-class baseline (always predict '1.5b'): {majority_baseline_acc:.3f}")
    if n_predicted_3b == 0:
        print(
            "  WARNING: the classifier never once predicted '3b' - it has "
            "collapsed to a majority-class predictor. Its accuracy above "
            "is indistinguishable from just always guessing the more "
            "common label, not evidence it learned a real routing signal "
            "from the embeddings. See README.md for what this does and "
            "doesn't tell us at this sample size."
        )

    print(f"\n{'=' * 64}")
    print("HEURISTIC ROUTER (router.py) ON THE SAME ROWS".center(64))
    print("=" * 64)
    heuristic_pred = heuristic_predictions(prompts)
    heuristic_acc = accuracy_score(y, heuristic_pred)
    heuristic_cm = confusion_matrix(y, heuristic_pred, labels=labels_order)
    print(f"  Heuristic accuracy: {heuristic_acc:.3f}")
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print(f"  {'':>10}{'pred 1.5b':>12}{'pred 3b':>10}")
    for i, row_label in enumerate(labels_order):
        print(f"  {'actual ' + row_label:>10}{heuristic_cm[i][0]:>12}{heuristic_cm[i][1]:>10}")

    print(f"\n{'=' * 64}")
    print("THE COMPARISON THAT ACTUALLY MATTERS".center(64))
    print("=" * 64)
    print(f"  Heuristic router accuracy:              {heuristic_acc:.3f}")
    print(f"  Trained classifier accuracy (CV):       {mean_acc:.3f}  (std: {std_acc:.3f})")
    print(f"  Majority-class baseline ('always 1.5b'): {majority_baseline_acc:.3f}")
    if n_predicted_3b == 0:
        print(
            "\n  -> The classifier's CV accuracy is not real evidence it beats the\n"
            "     heuristic: it collapsed to predicting the majority class every\n"
            "     time (see the WARNING above), and its accuracy is essentially\n"
            "     identical to the majority-class baseline. The honest reading of\n"
            "     this data is that 49 examples were not enough for logistic\n"
            "     regression on 384-dim embeddings to learn a routing signal -\n"
            "     not that the classifier is better than the heuristic."
        )
    elif mean_acc > heuristic_acc:
        print("  -> The trained classifier beats the heuristic on this data.")
    elif mean_acc < heuristic_acc:
        print("  -> The heuristic beats the trained classifier on this data.")
    else:
        print("  -> Dead even on this data.")
    print(
        "\n  Reported honestly either way - see README.md's methodology "
        "note on what ~49 examples can and can't tell us here."
    )

    # Final saved artifact is trained on ALL labeled rows, not one fold -
    # cross-validation above is purely for the accuracy estimate; the
    # deployed model should use every example available to it.
    final_model = LogisticRegression()
    final_model.fit(X, y)
    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(final_model, f)
    print(f"\nSaved final classifier (trained on all {len(prompts)} rows) to {CLASSIFIER_PATH}")


if __name__ == "__main__":
    main()
