"""Walk-forward machine learning with purging + embargo (Lopez de Prado).

DirectionModel  - two LightGBM classifiers estimating P(long trade wins) and
                  P(short trade wins) under the real trade-management rules.
MetaModel       - meta-labelling: given the desk's proposed side, estimate the
                  probability that *this* trade ends with positive R. Used to
                  veto weak signals and to scale size.

Every prediction for bar t comes from a model trained only on trades that had
fully *exited* before the start of t's fold, minus an embargo gap.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

LGB_PARAMS = dict(n_estimators=350, learning_rate=0.03, num_leaves=15, min_child_samples=80,
                  subsample=0.7, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
                  verbose=-1, random_state=7)


def _model():
    if lgb is not None:
        return lgb.LGBMClassifier(**LGB_PARAMS)
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                          min_samples_leaf=80, l2_regularization=5.0, random_state=7)


@dataclass
class FoldReport:
    start: pd.Timestamp
    end: pd.Timestamp
    n_train: int
    auc: float
    brier: float
    base_rate: float


@dataclass
class WFResult:
    pred: pd.DataFrame
    folds: list = field(default_factory=list)

    def summary(self) -> dict:
        if not self.folds:
            return {}
        aucs = [f.auc for f in self.folds if np.isfinite(f.auc)]
        return {"folds": len(self.folds), "mean_auc": float(np.mean(aucs)) if aucs else np.nan,
                "min_auc": float(np.min(aucs)) if aucs else np.nan,
                "mean_brier": float(np.mean([f.brier for f in self.folds]))}


def _fold_bounds(n: int, initial_frac: float, step: int):
    start = int(n * initial_frac)
    while start < n:
        yield start, min(n, start + step)
        start += step


def walk_forward(X: pd.DataFrame, y: pd.Series, exit_idx: np.ndarray, mask_train: np.ndarray,
                 initial_frac: float, step: int, embargo: int, sub: int = 2) -> WFResult:
    """Generic purged walk-forward binary classifier. Returns OOS P(y=1)."""
    n = len(X)
    pred = np.full(n, np.nan)
    folds = []
    pos = np.arange(n)
    Xv = X.values.astype(np.float32)
    yv = y.values
    for s, e in _fold_bounds(n, initial_frac, step):
        tr = mask_train & np.isfinite(yv) & (exit_idx >= 0) & (exit_idx < s - embargo) & (pos % sub == 0)
        if tr.sum() < 300 or len(np.unique(yv[tr])) < 2:
            continue
        m = _model()
        m.fit(Xv[tr], yv[tr].astype(int))
        pred[s:e] = m.predict_proba(Xv[s:e])[:, 1]
        te = np.arange(s, e)
        te = te[np.isfinite(yv[te]) & mask_train[te]]
        auc = brier = np.nan
        if len(te) > 30 and len(np.unique(yv[te])) == 2:
            auc = roc_auc_score(yv[te].astype(int), pred[te])
            brier = brier_score_loss(yv[te].astype(int), pred[te])
        folds.append(FoldReport(X.index[s], X.index[e - 1], int(tr.sum()), auc, brier,
                                float(np.nanmean(yv[tr]))))
    return WFResult(pd.Series(pred, index=X.index).to_frame("p"), folds)


def fit_full(X: pd.DataFrame, y: pd.Series, exit_idx: np.ndarray, mask_train: np.ndarray, sub: int = 2):
    """Model trained on everything with a completed label (for live use)."""
    n = len(X)
    pos = np.arange(n)
    yv = y.values
    tr = mask_train & np.isfinite(yv) & (exit_idx >= 0) & (pos % sub == 0)
    if tr.sum() < 300 or len(np.unique(yv[tr])) < 2:
        return None
    m = _model()
    m.fit(X.values[tr].astype(np.float32), yv[tr].astype(int))
    return m


class DirectionModel:
    def __init__(self, cols: list[str]):
        self.cols = cols
        self.reports: dict = {}
        self.models: dict = {}

    def walk_forward(self, f: pd.DataFrame, labels: pd.DataFrame, cfg) -> pd.DataFrame:
        X = f[self.cols]
        ok = np.ones(len(f), dtype=bool)
        out = pd.DataFrame(index=f.index)
        for side in ("long", "short"):
            y = (labels[f"r_{side}"] > 0).astype(float).where(labels[f"r_{side}"].notna())
            res = walk_forward(X, y, labels[f"exit_{side}"].values, ok, cfg.wf_initial_frac,
                               cfg.wf_step_bars, cfg.embargo_bars)
            out[f"ml_p_{side}"] = res.pred["p"]
            self.reports[side] = res
        out["ml_score"] = (2.5 * (out["ml_p_long"] - out["ml_p_short"])).clip(-1, 1).fillna(0.0)
        return out

    def fit_and_predict_last(self, f: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
        X = f[self.cols]
        ok = np.ones(len(f), dtype=bool)
        out = pd.DataFrame(index=f.index)
        for side in ("long", "short"):
            y = (labels[f"r_{side}"] > 0).astype(float).where(labels[f"r_{side}"].notna())
            m = fit_full(X, y, labels[f"exit_{side}"].values, ok)
            self.models[side] = m
            p = np.full(len(f), np.nan)
            if m is not None:
                p[-200:] = m.predict_proba(X.values[-200:].astype(np.float32))[:, 1]
            out[f"ml_p_{side}"] = p
        out["ml_score"] = (2.5 * (out["ml_p_long"] - out["ml_p_short"])).clip(-1, 1).fillna(0.0)
        return out

    def importance(self, top: int = 15) -> list[tuple[str, float]]:
        m = self.models.get("long")
        if m is None or not hasattr(m, "feature_importances_"):
            return []
        imp = pd.Series(m.feature_importances_, index=self.cols)
        if self.models.get("short") is not None:
            imp = imp + pd.Series(self.models["short"].feature_importances_, index=self.cols)
        imp = imp.sort_values(ascending=False).head(top)
        return [(k, float(v)) for k, v in imp.items()]


def meta_matrix(f: pd.DataFrame, ens: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    side = np.sign(ens["composite"]).replace(0, 1)
    X = f[cols].copy()
    for c in ens.columns:
        if c.startswith("a_"):
            X[c] = ens[c] * side          # analyst view *relative to the proposed side*
    X["composite_abs"] = ens["composite"].abs()
    X["agreement"] = ens["agreement"]
    X["trend_strength"] = ens["trend_strength"]
    X["side"] = side
    X["comp_vs_thr"] = (ens["composite"].abs() / ens["threshold"].replace(np.inf, np.nan)).fillna(0)
    return X


class MetaModel:
    def __init__(self, cols: list[str]):
        self.cols = cols
        self.report: WFResult | None = None
        self.model = None

    def _targets(self, ens: pd.DataFrame, labels: pd.DataFrame):
        side = np.sign(ens["composite"]).replace(0, 1)
        r = np.where(side > 0, labels["r_long"], labels["r_short"])
        ex = np.where(side > 0, labels["exit_long"], labels["exit_short"]).astype(int)
        y = pd.Series(np.where(np.isfinite(r), (r > 0).astype(float), np.nan), index=ens.index)
        q = ens["composite"].abs().rolling(1500, min_periods=300).quantile(0.6).shift(1)
        mask = (ens["composite"].abs() > q.fillna(np.inf)).values
        return y, ex, mask

    def walk_forward(self, f, ens, labels, cfg) -> pd.Series:
        X = meta_matrix(f, ens, self.cols)
        y, ex, mask = self._targets(ens, labels)
        res = walk_forward(X, y, ex, mask, cfg.wf_initial_frac, cfg.wf_step_bars, cfg.embargo_bars, sub=1)
        self.report = res
        return res.pred["p"].rename("meta_p")

    def fit_and_predict_last(self, f, ens, labels) -> pd.Series:
        X = meta_matrix(f, ens, self.cols)
        y, ex, mask = self._targets(ens, labels)
        self.model = fit_full(X, y, ex, mask, sub=1)
        p = np.full(len(f), np.nan)
        if self.model is not None:
            p[-200:] = self.model.predict_proba(X.values[-200:].astype(np.float32))[:, 1]
        return pd.Series(p, index=f.index, name="meta_p")
