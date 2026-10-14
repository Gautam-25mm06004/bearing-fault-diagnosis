"""Evaluation protocols.

The headline lesson of this project: how you split CWRU data decides the score.
* ``random_window_split`` mixes windows from the same 10-second recording into
  train and test. Neighbouring windows are near-duplicates, so this is leakage.
* ``leave_one_load_out`` trains on three motor loads and tests on the unseen
  fourth, so every test window comes from a recording the model never saw.
This mirrors the cross-load setting used for WDCNN by Zhang et al. (2017) and the
leakage critique in recent CWRU benchmarking work (see README references).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


def _score(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def random_window_split(model, X: pd.DataFrame, y: pd.Series, seed: int = 0) -> dict:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    m = clone(model).fit(Xtr, ytr)
    return _score(yte, m.predict(Xte))


LOCATION = {"Normal": "Normal", "B": "Ball", "IR": "InnerRace", "OR": "OuterRace"}
SIZES = ["007", "014", "021"]


def location_of(label: str) -> str:
    if label == "Normal":
        return "Normal"
    return LOCATION["IR" if label.startswith("IR") else "OR" if label.startswith("OR") else "B"]


def leave_one_size_out(model, X: pd.DataFrame, y: pd.Series, loads: pd.Series):
    """4-class fault-location task, tested on a fault size (= a physically different
    damaged bearing) never seen in training. Bearing-wise splitting is what recent
    CWRU benchmarking papers recommend; it is much harder than any window or load split.
    Normal data come from one healthy bearing, so its recordings are split by load:
    fold k tests on Normal at load k and trains on the other loads.
    """
    loc = y.map(location_of)
    size = y.str.extract(r"(\d{3})")[0]
    labels = ["Normal", "Ball", "InnerRace", "OuterRace"]
    per_fold, cm_total = [], np.zeros((4, 4), dtype=int)
    for k, held in enumerate(SIZES):
        is_norm = loc == "Normal"
        te = (size == held) | (is_norm & (loads == k))
        tr = ((size != held) & ~is_norm) | (is_norm & (loads != k))
        m = clone(model).fit(X[tr], loc[tr])
        pred = m.predict(X[te])
        s = _score(loc[te], pred)
        s["test_fault_size_mils"] = held
        per_fold.append(s)
        cm_total += confusion_matrix(loc[te], pred, labels=labels)
    df = pd.DataFrame(per_fold)
    return ({"accuracy_mean": float(df.accuracy.mean()), "accuracy_min": float(df.accuracy.min()),
             "macro_f1_mean": float(df.macro_f1.mean())}, df, cm_total, labels)


def leave_one_load_out(model, X: pd.DataFrame, y: pd.Series, loads: pd.Series, labels: list[str]):
    per_fold, cm_total = [], np.zeros((len(labels), len(labels)), dtype=int)
    for held in sorted(loads.unique()):
        tr, te = loads != held, loads == held
        m = clone(model).fit(X[tr], y[tr])
        pred = m.predict(X[te])
        s = _score(y[te], pred)
        s["test_load_hp"] = int(held)
        per_fold.append(s)
        cm_total += confusion_matrix(y[te], pred, labels=labels)
    df = pd.DataFrame(per_fold)
    summary = {
        "accuracy_mean": float(df.accuracy.mean()),
        "accuracy_min": float(df.accuracy.min()),
        "macro_f1_mean": float(df.macro_f1.mean()),
    }
    return summary, df, cm_total
