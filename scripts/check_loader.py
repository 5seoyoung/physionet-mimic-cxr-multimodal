import torch
from torch.utils.data import DataLoader
from dataloaders.multimodal import MIMICCXRMultimodal

def main():
    ds = MIMICCXRMultimodal(
        split_csv="data/meta.csv",
        img_root="data/images",
        labels_csv="data/labels.csv",
        tok_jsonl=None,
        reports_csv="data/reports.csv",  # CSV에서 바로 토큰화
        tokenizer_name="distilbert-base-uncased",
        image_size=224,
    )
    dl = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(dl))
    print("image:", batch["image"].shape)              # [B,3,224,224]
    print("input_ids:", batch["input_ids"].shape)      # [B, L]
    print("attention_mask:", batch["attention_mask"].shape)
    print("label:", batch["label"].shape)              # [B, 14]
    print("study_id:", batch["study_id"])

if __name__ == "__main__":
    main()
