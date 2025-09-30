# 간단 로드 & 단일 샘플 추론 예시 (필요 시 확장)
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer
from models.student.student_model import StudentImageEncoder
from models.fusion import LateFusionHead
from training.train_student import mean_pool

def load_models(text_model="distilbert-base-uncased"):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    img_enc = StudentImageEncoder(out_dim=1024); img_enc.load_state_dict(torch.load("weights/mobilenetv3_student.pth", map_location=device)); img_enc.eval().to(device)
    head = LateFusionHead(1024, 768, 14); head.load_state_dict(torch.load("weights/fusion_head.pth", map_location=device)); head.eval().to(device)
    txt = AutoModel.from_pretrained(text_model); txt.load_state_dict(torch.load("weights/distilbert_student.bin", map_location=device)); txt.eval().to(device)
    tok = AutoTokenizer.from_pretrained(text_model)
    return img_enc, head, txt, tok, device

def predict(png_path, report_text):
    img_enc, head, txt, tok, device = load_models()
    tx = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                             transforms.Lambda(lambda t: t.expand(3, *t.shape[1:]) if t.shape[0]==1 else t),
                             transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    img = tx(Image.open(png_path).convert("L")).unsqueeze(0).to(device)
    enc = tok(report_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        img_feat = img_enc(img)
        out = txt(**enc)
        txt_feat = mean_pool(out.last_hidden_state, enc["attention_mask"])
        logits = head(img_feat, txt_feat)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
    return probs

if __name__ == "__main__":
    p = predict("data/images/10000001.png", "No acute cardiopulmonary abnormality.")
    print("Pred probs (14):", p)
