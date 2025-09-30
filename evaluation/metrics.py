
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def multilabel_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    C = y_true.shape[1]
    aurocs, auprcs, f1s = [], [], []
    valid = 0
    for c in range(C):
        yc = y_true[:, c]
        pc = y_prob[:, c]
        # 한 클래스만 있으면 스킵
        if yc.max() == yc.min():
            continue
        valid += 1
        try:
            aurocs.append(roc_auc_score(yc, pc))
        except ValueError:
            pass
        try:
            auprcs.append(average_precision_score(yc, pc))
        except ValueError:
            pass
        yhat = (pc >= thr).astype(int)
        f1s.append(f1_score(yc, yhat, zero_division=0))
    if valid == 0:
        return {"AUROC": float("nan"), "AUPRC": float("nan"), "F1": 0.0, "valid_classes": 0}
    return {
        "AUROC": float(np.mean(aurocs)) if aurocs else float("nan"),
        "AUPRC": float(np.mean(auprcs)) if auprcs else float("nan"),
        "F1":    float(np.mean(f1s))    if f1s    else 0.0,
        "valid_classes": valid
    }

def best_global_threshold(y_true, y_prob, grid=None):
    """글로벌 단일 임계값으로 micro-F1 최적화"""
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_t, best_micro = 0.5, 0.0
    for t in grid:
        yhat = (y_prob >= t).astype(int)
        micro = f1_score(y_true, yhat, average="micro", zero_division=0)
        if micro > best_micro:
            best_micro, best_t = micro, t
    return float(best_t), float(best_micro)
