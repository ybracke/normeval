import normeval
import pytest


def test_cer() -> None:
    dists = [0, 1, 2, 3, 4]
    lens = [2, 4, 5, 6, 7]
    assert normeval.cer(dists, lens) == [0, 0.25, 0.4, 0.5, (4 / 7)]


# Zero division should fail, as strings of length 0 shouldn't occur in the data
# and the user should be pointed to them
def test_cer_divideby0() -> None:
    dists = [
        1,
    ]
    lens = [
        0,
    ]
    with pytest.raises(ZeroDivisionError):
        _ = normeval.cer(dists, lens)


def test_seqsdist() -> None:
    s1 = ["AN", "diesem", "fünften", "Stucke", "des", "puchs", "von"]
    s2 = ["An", "diesem", "fünften", "Stück", "des", "Buchs", ""]
    dists = normeval.seqsdist(
        s1,
        s2,
    )
    assert dists == [1, 0, 0, 2, 0, 1, 3]


def test_create_entry() -> None:
    vals = [1, 5.4, 4, 9.6]
    entry = normeval.eval._create_entry(vals)
    assert entry == {
        "mean": 5,
        "std": 3.095157508108432,
        "max": 9.6,
        "min": 1,
        "median": 4.7,
        "q1,q2,q3,q4": [3.25, 4.7, 6.45, 9.6],
    }


def test_evaluate_emptylists() -> None:
    assert normeval.evaluate([], []) == {}


def test_evaluate_difflengths() -> None:
    s1 = ["AN", "diesem", "fünften", "Stucke", "des", "puchs", "von"]
    s2 = [
        "An",
    ]
    with pytest.raises(ValueError, match=r"Sequences must be of equal length.*"):
        _ = normeval.evaluate(s1, s2)


def test_evaluate_some() -> None:
    pred = ["AN", "diesem", "Stu\u0364cke", "des", "puchs", "fon"]
    gold = ["An", "diesem", "Stück", "des", "Buchs", "von"]
    results = normeval.evaluate(pred, gold, ("LD", "acc", "macro-f1"))
    assert normeval.seqsdist(pred, gold) == [1, 0, 3, 0, 1, 1]
    assert results["LD"]["mean"] == 1
    assert results["LD"]["std"] == 1
    assert results["LD"]["max"] == 3
    assert results["LD"]["min"] == 0
    assert results["LD"]["median"] == 1
    assert results["LD"]["q1,q2,q3,q4"] == [0.25, 1, 1, 3]
    assert results["acc"] == 2 / 6
    # macro-f1: 10 classes , 2 x F1=1.0; 8x F1 = 0 -> 2/10
    assert results["macro-f1"] == 0.2  