import argparse, pathlib, torch, torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel
from dataloaders.multimodal import MIMICCXRMultimodal
from models.student.student_model import StudentImageEncoder
from models.fusion import LateFusionHead
from training.losses import KDLoss

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    s = (last_hidden_state * mask).sum(dim=1)
    d = mask.sum(dim=1).clamp(min=1e-6)
    return s / d

def build_text_encoder(name="distilbert-base-uncased", trainable=False):
    model = AutoModel.from_pretrained(name)
    for p in model.parameters(): p.requires_grad = trainable
    return model


@torch.no_grad()
def evaluate(dl, img_enc, txt_enc, head, device):
    import numpy as np
    from evaluation.metrics import multilabel_metrics
    img_enc.eval(); head.eval(); txt_enc.eval()
    probs_all, y_all = [], []
    for batch in dl:
        imgs = batch["image"].to(device)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["label"].cpu().numpy()
        img_feat = img_enc(imgs)
        out = txt_enc(input_ids=ids, attention_mask=mask)
        txt_feat = mean_pool(out.last_hidden_state, mask)
        logits = head(img_feat, txt_feat)
        probs = torch.sigmoid(logits).cpu().numpy()
        probs_all.append(probs); y_all.append(y)
    probs_all = np.vstack(probs_all); y_all = np.vstack(y_all)
    return multilabel_metrics(y_all, probs_all, thr=0.5)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", default="data/meta.csv")
    ap.add_argument("--img_root", default="data/images")
    ap.add_argument("--labels_csv", default="data/labels.csv")
    ap.add_argument("--reports_csv", default="data/reports.csv")
    ap.add_argument("--val_split_csv", default=None)
    ap.add_argument("--tok_jsonl", default=None)
    ap.add_argument("--text_model", default="distilbert-base-uncased")
    ap.add_argument("--unfreeze_text", action="store_true")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--num_labels", type=int, default=14)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--use_kd", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=3.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    ds = MIMICCXRMultimodal(
        split_csv=args.split_csv, img_root=args.img_root, labels_csv=args.labels_csv,
        tok_jsonl=args.tok_jsonl, reports_csv=args.reports_csv,
        tokenizer_name=args.text_model, image_size=224
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

# validation loader (optional)
if args.val_split_csv:
    from dataloaders.multimodal import MIMICCXRMultimodal
    ds_val = MIMICCXRMultimodal(
        split_csv=args.val_split_csv, img_root=args.img_root, labels_csv=args.labels_csv,
        tok_jsonl=args.tok_jsonl, reports_csv=args.reports_csv,
        tokenizer_name=args.text_model, image_size=224
    )
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
else:
    dl_val = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    img_enc = StudentImageEncoder(out_dim=1024).to(device)
    txt_enc = build_text_encoder(args.text_model, trainable=args.unfreeze_text).to(device)
    head = LateFusionHead(1024, 768, args.num_labels).to(device)

    params = list(img_enc.parameters()) + list(head.parameters())
    if args.unfreeze_text: params += list(txt_enc.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    kd = KDLoss(alpha=args.alpha, temperature=args.temperature) if args.use_kd else None

    for ep in range(1, args.epochs+1):
        img_enc.train(); head.train(); 
        if any(p.requires_grad for p in txt_enc.parameters()): txt_enc.train()
        bce = torch.nn.BCEWithLogitsLoss()
        total, n = 0.0, 0
        for batch in dl:
            imgs = batch["image"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)

            img_feat = img_enc(imgs)
            out = txt_enc(input_ids=ids, attention_mask=mask)
            txt_feat = mean_pool(out.last_hidden_state, mask)
            logits = head(img_feat, txt_feat)

            loss = bce(logits, y) if kd is None else kd(logits, logits.detach(), y)  # KD teacher dummy
            optim.zero_grad(); loss.backward(); optim.step()

            total += loss.item() * imgs.size(0); n += imgs.size(0)

        metrics = evaluate(dl, img_enc, txt_enc, head, device)
        print(f"[Epoch {ep}] loss={total/max(n,1):.4f}  AUROC={metrics['AUROC']:.4f}  AUPRC={metrics['AUPRC']:.4f}  F1={metrics['F1']:.4f}  best_thr={metrics.get('best_thr',0):.2f}  F1_micro@best={metrics.get('F1_micro@best_thr',0):.4f}")

    pathlib.Path("weights").mkdir(parents=True, exist_ok=True)
    torch.save(img_enc.state_dict(), "weights/mobilenetv3_student.pth")
    torch.save(head.state_dict(), "weights/fusion_head.pth")
    torch.save(txt_enc.state_dict(), "weights/distilbert_student.bin")
    print("Saved weights to weights/")

if __name__ == "__main__":
    main()
