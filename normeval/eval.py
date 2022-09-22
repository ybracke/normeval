import Levenshtein
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from typing import List, Union, Iterable, Dict, Callable


def evaluate(
    pred: List[str], gold: List[str], methods: Union[str, Iterable[str]] = "all"
) -> Dict:
    """
    Create a dictionary with evaluation results for predicted and gold labels.

    Arguments
    ---------
    `methods` : either 'all' or an Iterable containing one or more of the
    following: `'LD', 'CER', 'acc', 'macro-f1'`

    Returns
    --------
    `results` looks like this: ``` {
        "ld": {
            "mean": 0.21266331658291457, "std": 0.78874013286826, "max": 8,
            "min": 0, "median": 0.0, "q1,q2,q3,q4": [
                0.0, 0.0, 0.0, 8.0
            ]
        }, "CER" : {
            ### as above ###
        }, "accuracy": 0.893467336683417, "macro-F1": 0.7188353353436098
    }
    ```
    """

    if methods == "all":
        methods = ["LD", "CER", "acc", "macro-f1"]

    if len(pred) != len(gold):
        raise ValueError(f"Sequences must be of equal length: {len(pred)}!={len(gold)}")

    # Both lists empty?
    if len(pred) == 0:
        return {}

    # Create results dictionary
    results = {}

    # Compute string distances if needed
    if any(m in methods for m in ["LD", "CER"]):
        dists = seqsdist(pred, gold)

    if "LD" in methods:
        results["LD"] = _create_entry(dists)

    if "CER" in methods:
        cers = cer(dists, [len(w) for w in gold])
        results["CER"] = _create_entry(cers)

    if "acc" in methods:
        results["acc"] = accuracy_score(gold, pred)

    if "macro-f1" in methods:
        results["macro-f1"] = f1_score(gold, pred, average="macro", zero_division=0)

    return results


def seqsdist(
    seq1: List[str], seq2: List[str], dist_func: Callable = Levenshtein.distance
) -> List[int]:
    """
    String distances of elements in 2 lists at same index position

    `seq1`, `seq2` are lists of strings, must have same length
    """

    return [dist_func(w1, w2) for w1, w2 in zip(seq1, seq2)]


def cer(dists: List[int], lengths: List[int]) -> List[float]:
    """
    Character error rate (CER)
    Levenshtein distance normalized by reference word length
    """
    return [0 if dist == 0 else dist / l for dist, l in zip(dists, lengths)]


def _create_entry(values: List[float]) -> Dict["str", Union[float, List]]:
    """
    Create a dictionary with different function results for a numerical array.

    Functions are: np.mean, np.std, max, min, np.median, np.quantile

    Helper function for `evaluate`

    values : list/array of numbers, some string metric
    """
    entry = {}
    funcs = [np.mean, np.std, max, min, np.median]
    for func in funcs:
        entry[func.__name__] = func(values)
    # Add quantiles
    entry["q1,q2,q3,q4"] = list(np.quantile(values, [0.25, 0.5, 0.75, 1]))

    return entry
