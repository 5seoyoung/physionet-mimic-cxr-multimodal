
import argparse, os, numpy as np
from pathlib import Path
from PIL import Image
import pydicom

def _to_uint8(x):
    x = x.astype(np.float32)
    x -= x.min()
    if x.max() > 0:
        x = x / x.max()
    x = (x * 255.0).clip(0,255).astype(np.uint8)
    return x

def dcm_to_png(dcm_path, out_path):
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)

    # MONOCHROME1이면 반전
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        arr = arr.max() - arr

    # rescale
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    # uint8 normalize
    arr = _to_uint8(arr)
    Image.fromarray(arr).save(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="DICOM root directory")
    ap.add_argument("--dst", required=True, help="PNG output directory")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--map_csv", default=None, help="Optional CSV with columns: dicom,study_id")
    args = ap.parse_args()

    outdir = Path(args.dst); outdir.mkdir(parents=True, exist_ok=True)

    if args.map_csv:
        import pandas as pd
        df = pd.read_csv(args.map_csv)
        for _, row in df.iterrows():
            dcm = Path(row["dicom"])
            sid = str(row["study_id"])
            if not dcm.exists():
                print(f"[skip] missing {dcm}")
                continue
            tmp = outdir / f"{sid}.png"
            dcm_to_png(dcm, tmp)
            # resize
            img = Image.open(tmp).convert("L").resize((args.size, args.size))
            img.save(tmp)
    else:
        # 이름 유지 모드: <src>/.../name.dcm -> <dst>/name.png
        from PIL import Image
        for dcm in Path(args.src).rglob("*.dcm"):
            name = dcm.stem + ".png"
            out = outdir / name
            try:
                dcm_to_png(dcm, out)
                img = Image.open(out).convert("L").resize((args.size, args.size))
                img.save(out)
            except Exception as e:
                print(f"[error] {dcm}: {e}")

if __name__ == "__main__":
    main()
