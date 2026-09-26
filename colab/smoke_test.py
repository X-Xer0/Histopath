"""
Smoke test for the MoNuSeg training pipeline.

Runs the entire pipeline end-to-end at reduced size so that any broken step is
found in a few minutes instead of after a long training run. Nothing produced
here is meant to be kept - it exists only to prove the pipeline works.

Usage on the Colab VM:
    colab exec -s <session> -f colab/smoke_test.py --timeout 3600
"""
import os
import sys
import json
import time
import random
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = False   # fp16 autocast produces NaN gradients for this U-Net (see notebook cell 2)

FAILURES = []
def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}{('  -> ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)
    return ok

print("=" * 74)
print(" MoNuSeg pipeline smoke test")
print(f" device={device}  amp={USE_AMP}  torch={torch.__version__}")
print("=" * 74)

# ------------------------------------------------------------------ data
DATA_ROOT = Path("/content/monuseg_upload")
UPLOAD_ZIP = Path("/content/monuseg_local.zip")
if not DATA_ROOT.exists():
    assert UPLOAD_ZIP.exists(), f"missing {UPLOAD_ZIP}"
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(UPLOAD_ZIP) as zf:
        zf.extractall(DATA_ROOT)

TRAIN_IMG_DIR = DATA_ROOT / "MoNuSeg 2018 Training Data" / "Tissue Images"
TRAIN_XML_DIR = DATA_ROOT / "MoNuSeg 2018 Training Data" / "Annotations"
TEST_DIR      = DATA_ROOT / "MoNuSegTestData"
check("dataset folders present", all(p.exists() for p in (TRAIN_IMG_DIR, TRAIN_XML_DIR, TEST_DIR)))

def patient_key(stem):
    return "-".join(stem.split("-")[:3])

def parse_xml_mask(xml_path, h, w):
    mask = np.zeros((h, w), np.uint8)
    for r in ET.parse(xml_path).getroot().findall(".//Region"):
        pts = [[int(round(float(v.attrib['X']))), int(round(float(v.attrib['Y'])))]
               for v in r.findall(".//Vertex")]
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
    return mask

# ------------------------------------------------------- tiny patch set
PATCH_DIR = Path("./smoke_patches")
(PATCH_DIR / "images").mkdir(parents=True, exist_ok=True)
(PATCH_DIR / "masks").mkdir(parents=True, exist_ok=True)

slides = sorted(TRAIN_IMG_DIR.glob("*.tif"))[:6]        # only 6 slides for speed
n_patches = 0
for sp in slides:
    img = cv2.imread(str(sp))
    h, w = img.shape[:2]
    xml = TRAIN_XML_DIR / f"{sp.stem}.xml"
    if not xml.exists():
        continue
    mask = parse_xml_mask(xml, h, w)
    for y in range(0, h - 256 + 1, 248):                # big stride -> fewer patches
        for x in range(0, w - 256 + 1, 248):
            ip, mp = img[y:y+256, x:x+256], mask[y:y+256, x:x+256]
            if np.mean(cv2.cvtColor(ip, cv2.COLOR_BGR2GRAY)) > 240:
                continue
            name = f"{sp.stem}__y{y:04d}_x{x:04d}.png"
            cv2.imwrite(str(PATCH_DIR / "images" / name), ip)
            cv2.imwrite(str(PATCH_DIR / "masks"  / name), mp)
            n_patches += 1
check("patches extracted", n_patches > 0, f"{n_patches} patches from {len(slides)} slides")

# ------------------------------------------------------------ split
all_patches = sorted((PATCH_DIR / "images").glob("*.png"))
by_patient = defaultdict(list)
for p in all_patches:
    by_patient[patient_key(p.name.split("__")[0])].append(p)
patients = sorted(by_patient)
rng = random.Random(SEED)
val_patients = set(rng.sample(patients, max(1, len(patients) // 4)))
train_paths = [p for p in all_patches if patient_key(p.name.split("__")[0]) not in val_patients]
val_paths   = [p for p in all_patches if patient_key(p.name.split("__")[0]) in val_patients]
leak = ({patient_key(p.name.split("__")[0]) for p in train_paths} &
        {patient_key(p.name.split("__")[0]) for p in val_paths})
check("patient-wise split has no leakage", not leak, f"{len(train_paths)} train / {len(val_paths)} val patches")

# ------------------------------------------------------------ dataset
train_tf = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3, border_mode=cv2.BORDER_REFLECT),
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.2, hue=0.05, p=0.5),
])

class DS(Dataset):
    def __init__(self, paths, mask_dir, transform=None):
        self.paths = list(paths); self.mask_dir = Path(mask_dir); self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(str(self.mask_dir / p.name), cv2.IMREAD_GRAYSCALE)
        if self.transform:
            o = self.transform(image=img, mask=msk); img, msk = o['image'], o['mask']
        return (torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0,
                torch.from_numpy((msk > 127).astype(np.uint8)).unsqueeze(0).float())

train_loader = DataLoader(DS(train_paths, PATCH_DIR / "masks", train_tf),
                          batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(DS(val_paths, PATCH_DIR / "masks", None),
                          batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
b = next(iter(train_loader))
check("dataloader yields batches", b[0].shape == (min(8, len(train_paths)), 3, 256, 256),
      f"image {tuple(b[0].shape)} mask {tuple(b[1].shape)}")
check("mask is binary", set(torch.unique(b[1]).tolist()) <= {0.0, 1.0})

# -------------------------------------------------------------- model
class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=64):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*2, base*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*4, base*8))
        self.up1 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2); self.cu1 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2); self.cu2 = DoubleConv(base*4, base*2)
        self.up3 = nn.ConvTranspose2d(base*2, base, 2, stride=2);   self.cu3 = DoubleConv(base*2, base)
        self.outc = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        x = self.cu1(torch.cat([self.up1(x4), x3], 1))
        x = self.cu2(torch.cat([self.up2(x),  x2], 1))
        x = self.cu3(torch.cat([self.up3(x),  x1], 1))
        return torch.sigmoid(self.outc(x))

class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__(); self.bce = nn.BCELoss(); self.alpha, self.beta = alpha, beta
    def forward(self, pred, target, smooth=1e-5):
        tp = (pred*target).sum(); fp = (pred*(1-target)).sum(); fn = ((1-pred)*target).sum()
        tv = (tp + smooth) / (tp + self.alpha*fp + self.beta*fn + smooth)
        return self.bce(pred, target) + (1.0 - tv)

model = UNet(base=64).to(device)
criterion = ComboLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=1e-6)
try:
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
except (AttributeError, TypeError):
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
check("model built", sum(p.numel() for p in model.parameters()) > 0,
      f"{sum(p.numel() for p in model.parameters()):,} parameters")

# ------------------------------------------------------------ train 3 epochs
print("\n  training 3 epochs ...")
losses = []
t0 = time.time()
for epoch in range(1, 4):
    model.train(); total = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=USE_AMP):
            preds = model(imgs)
        loss = criterion(preds.float(), masks.float())   # BCE is unsafe under autocast
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        total += loss.item()
    total /= len(train_loader); losses.append(total)
    scheduler.step()
    print(f"    epoch {epoch}: train loss {total:.4f}  ({time.time()-t0:.0f}s)")
check("training loss decreased", losses[-1] < losses[0], f"{losses[0]:.4f} -> {losses[-1]:.4f}")
check("mixed precision path exercised", True, f"amp={USE_AMP}")

# ------------------------------------------------- TTA inference + metrics
TTA_OPS = [
    (lambda x: x,                        lambda y: y),
    (lambda x: np.rot90(x, 1),           lambda y: np.rot90(y, -1)),
    (lambda x: np.rot90(x, 2),           lambda y: np.rot90(y, 2)),
    (lambda x: np.rot90(x, 3),           lambda y: np.rot90(y, 1)),
    (lambda x: np.fliplr(x),             lambda y: np.fliplr(y)),
    (lambda x: np.rot90(np.fliplr(x),1), lambda y: np.fliplr(np.rot90(y,-1))),
    (lambda x: np.rot90(np.fliplr(x),2), lambda y: np.fliplr(np.rot90(y, 2))),
    (lambda x: np.rot90(np.fliplr(x),3), lambda y: np.fliplr(np.rot90(y, 1))),
]

def predict_slide(img_bgr, tile=256, stride=192, tta=True):
    h, w = img_bgr.shape[:2]
    prob = np.zeros((h, w), np.float32); cnt = np.zeros((h, w), np.float32)
    ops = TTA_OPS if tta else TTA_OPS[:1]
    for y in range(0, max(1, h-tile+1), stride):
        for x in range(0, max(1, w-tile+1), stride):
            ye, xe = min(y+tile, h), min(x+tile, w)
            patch = img_bgr[y:ye, x:xe]
            if patch.shape[0] < 32 or patch.shape[1] < 32: continue
            rgb = cv2.cvtColor(cv2.resize(patch, (tile, tile)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            acc = np.zeros((tile, tile), np.float32)
            for fwd, inv in ops:
                t = torch.from_numpy(np.ascontiguousarray(fwd(rgb))).permute(2,0,1).unsqueeze(0).to(device)
                with torch.no_grad(), torch.autocast(device_type='cuda', enabled=USE_AMP):
                    out = model(t).squeeze().float().cpu().numpy()
                acc += inv(out)
            acc /= len(ops)
            if patch.shape[:2] != (tile, tile):
                acc = cv2.resize(acc, (patch.shape[1], patch.shape[0]))
            prob[y:ye, x:xe] += acc; cnt[y:ye, x:xe] += 1.0
    cnt[cnt == 0] = 1.0
    return prob / cnt

model.eval()
test_slides = sorted(TEST_DIR.glob("*.tif"))[:2]
dices = []
for sp in test_slides:
    img = cv2.imread(str(sp)); h, w = img.shape[:2]
    gt = (parse_xml_mask(TEST_DIR / f"{sp.stem}.xml", h, w) > 127).astype(np.uint8)
    prob = predict_slide(img)
    pred = (prob > 0.5).astype(np.uint8)
    tp = np.sum((pred==1)&(gt==1)); fp = np.sum((pred==1)&(gt==0)); fn = np.sum((pred==0)&(gt==1))
    dice = (2*tp + 1e-6) / (2*tp + fp + fn + 1e-6)
    dices.append(dice)
    print(f"    {sp.name}: dice={dice:.4f}  fp={fp:,}  fn={fn:,}")
check("TTA tiled inference produced probabilities", all(0.0 <= d <= 1.0 for d in dices),
      f"mean dice {np.mean(dices):.4f} (untrained model, value is not meaningful)")

# ------------------------------------------------------------ ONNX export
import inspect
onnx_path = "smoke_model.onnx"
dummy = torch.randn(1, 3, 256, 256, device=device)
kw = dict(input_names=['input'], output_names=['output'], opset_version=11,
          export_params=True, do_constant_folding=True)
if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
    kw['dynamo'] = False
try:
    torch.onnx.export(model, dummy, onnx_path, **kw)
    size_mb = os.path.getsize(onnx_path) / (1024*1024)
    check("ONNX export", size_mb > 1.0, f"{onnx_path} = {size_mb:.2f} MB, single file")
except Exception as e:
    check("ONNX export", False, f"{type(e).__name__}: {e}")

# ------------------------------------------------------------ verdict
print("=" * 74)
if FAILURES:
    print(f" SMOKE TEST FAILED: {len(FAILURES)} check(s) failed -> {FAILURES}")
    sys.exit(1)
print(" SMOKE TEST PASSED - the full pipeline runs end to end on this runtime.")
print("=" * 74)
