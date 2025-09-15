import os, json, torch
from torch.utils.data import Dataset
from PIL import Image

class ReportsIndex:
    def __init__(self, jsonl_path=None, csv_path=None, tokenizer=None, max_len=128):
        self.map = {}
        if jsonl_path and os.path.exists(jsonl_path):
            with open(jsonl_path) as f:
                for line in f:
                    obj = json.loads(line)
                    self.map[str(obj["study_id"])] = (obj["input_ids"], obj["attention_mask"])
        elif csv_path and tokenizer:
            # on-the-fly tokenization from CSV (columns: study_id,report)
            import pandas as pd
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                enc = tokenizer(
                    str(row["report"]),
                    padding="max_length", truncation=True, max_length=max_len
                )
                self.map[str(row["study_id"])] = (enc["input_ids"], enc["attention_mask"])
        else:
            raise ValueError("Provide either jsonl_path or (csv_path+tokenizer).")

class MIMICCXRMultimodal(Dataset):
    def __init__(self, split_csv, img_root, labels_csv, tok_jsonl=None, reports_csv=None,
                 tokenizer_name="distilbert-base-uncased", image_size=224):
        """
        split_csv : CSV [patient_id, study_id]
        img_root  : PNG root (expects {study_id}.png)
        labels_csv: CSV [study_id,label_0...label_13]
        tok_jsonl : tokenized reports (optional)
        reports_csv: raw reports CSV (optional, used if tok_jsonl is None)
        """
        import pandas as pd
        from torchvision import transforms
        from transformers import AutoTokenizer

        self.df = pd.read_csv(split_csv)
        self.labels = pd.read_csv(labels_csv).set_index("study_id")
        self.img_root = img_root
        self.tx = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # grayscale → 3ch
            transforms.Lambda(lambda t: t.expand(3, *t.shape[1:]) if t.shape[0]==1 else t),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

        # text
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.rep_idx = ReportsIndex(
            jsonl_path=tok_jsonl,
            csv_path=reports_csv,
            tokenizer=tok,
            max_len=128
        )

        # label cleanup: replace -1 → 0
        self.labels = self.labels.applymap(lambda v: 0 if v == -1 else v)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        sid = str(row["study_id"])

        # image
        img_path = os.path.join(self.img_root, f"{sid}.png")
        img = Image.open(img_path).convert("L")
        img = self.tx(img)

        # text
        input_ids, attention_mask = self.rep_idx.map[sid]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # labels
        y = torch.tensor(self.labels.loc[int(sid)].values, dtype=torch.float32)

        return {
            "image": img,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": y,
            "study_id": sid
        }
