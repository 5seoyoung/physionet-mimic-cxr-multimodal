import argparse, pandas as pd, numpy as np, pathlib

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", required=True)  # columns: patient_id,study_id
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.meta_csv)
    patients = df["patient_id"].unique().tolist()

    rng = np.random.default_rng(args.seed)
    rng.shuffle(patients)

    n = len(patients)
    # 안전한 분할: 매우 작은 n에서도 train/val/test가 비지 않도록
    if n == 0:
        raise SystemExit("No patients found in meta.csv")
    elif n == 1:
        tr_pat, va_pat, te_pat = patients[:1], [], []
    elif n == 2:
        tr_pat, va_pat, te_pat = patients[:1], patients[1:2], []
    elif n == 3:
        tr_pat, va_pat, te_pat = patients[:1], patients[1:2], patients[2:3]
    else:
        tr = max(1, int(round(0.8 * n)))
        va = max(1, int(round(0.1 * n)))
        # test가 최소 1이 되도록 보정
        if tr + va >= n:
            va = max(1, min(va, n - tr - 1))
            if tr + va >= n:
                tr = max(1, n - va - 1)
        tr_pat = patients[:tr]
        va_pat = patients[tr:tr+va]
        te_pat = patients[tr+va:]

    df["split"] = "test"  # 기본값
    df.loc[df["patient_id"].isin(tr_pat), "split"] = "train"
    df.loc[df["patient_id"].isin(va_pat), "split"] = "val"

    out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s][["patient_id", "study_id"]]
        sub.to_csv(out / f"{s}.csv", index=False)
        print(f"{s}: {len(sub)} rows, {sub['patient_id'].nunique()} patients")
