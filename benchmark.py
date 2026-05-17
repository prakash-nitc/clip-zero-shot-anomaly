"""
MVTec AD Benchmark — CLIP Zero-Shot vs. One-Class SVM
======================================================
Reports three things on the same dataset / same CLIP backbone:

  1. Image-level AUROC : CLIP Zero-Shot vs. trained One-Class SVM
  2. Pixel-level AUROC  : defect localization via windowed scoring
  3. Prompt ablation    : generic vs. category-specific vs. ensemble

Run on a GPU (Kaggle free T4 works):
    python benchmark.py --data /path/to/mvtec-ad

Dataset: https://kaggle.com/datasets/ipythonx/mvtec-ad
Note: pixel-level AUROC uses windowed inference over every test image
and is the slow part (~25-40 min on a T4). Image-level + ablation
finish in ~6 min.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────
CATEGORIES = [
    "carpet", "grid", "leather", "tile", "wood",
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper",
]
DISPLAY = {
    "carpet": "carpet", "grid": "grid", "leather": "leather", "tile": "tile",
    "wood": "wood", "bottle": "bottle", "cable": "cable", "capsule": "capsule",
    "hazelnut": "hazelnut", "metal_nut": "metal nut", "pill": "pill",
    "screw": "screw", "toothbrush": "toothbrush", "transistor": "transistor",
    "zipper": "zipper",
}
NORMAL_STATES   = ["good", "perfect", "flawless", "pristine", "normal", "unblemished"]
ABNORMAL_STATES = ["damaged", "defective", "broken", "flawed", "abnormal", "imperfect"]
TEMPLATES = [
    "a photo of a {state} {object}",
    "a {state} {object}",
    "a photo of a {state} {object} for quality inspection",
    "a close-up photo of a {state} {object}",
]

PIXEL_GRID = 8     # windows per axis for localization
PIXEL_RES  = 128   # resolution at which pixel-AUROC is computed


def build_prompts(category, states):
    obj = DISPLAY.get(category, category)
    return [t.format(state=s, object=obj) for t in TEMPLATES for s in states]


# ── Dataset ───────────────────────────────────────────────────────
class MVTecSplit(Dataset):
    """Images from one split (train/test) of a single category."""

    def __init__(self, root, category, split, transform):
        self.samples, self.transform = [], transform
        split_dir = Path(root) / category / split
        for subdir in sorted(split_dir.iterdir()):
            if not subdir.is_dir():
                continue
            label = 0 if subdir.name == "good" else 1
            for f in sorted(subdir.iterdir()):
                if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    self.samples.append((str(f), label))
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(Image.open(path).convert("RGB")), label


def list_test_with_masks(root, category):
    """Yield (image_path, mask_path_or_None) for every test image."""
    test_dir = Path(root) / category / "test"
    gt_dir   = Path(root) / category / "ground_truth"
    items = []
    for subdir in sorted(test_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for f in sorted(subdir.iterdir()):
            if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            if subdir.name == "good":
                items.append((str(f), None))
            else:
                mask = gt_dir / subdir.name / f"{f.stem}_mask.png"
                items.append((str(f), str(mask) if mask.exists() else None))
    return items


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True,              help="MVTec AD root")
    parser.add_argument("--model",      default="ViT-L-14")
    parser.add_argument("--pretrained", default="laion2b_s32b_b82k")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-pixel", action="store_true",
                        help="Skip the slow pixel-level AUROC stage")
    args = parser.parse_args()

    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, args.pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    print(f"Model  : {args.model} ({args.pretrained})\n")

    @torch.no_grad()
    def encode_text(prompts):
        tokens = tokenizer(prompts).to(device)
        feats  = F.normalize(model.encode_text(tokens), dim=-1)
        return F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)

    @torch.no_grad()
    def text_pair(normal_prompts, abnormal_prompts):
        return torch.cat([encode_text(normal_prompts),
                          encode_text(abnormal_prompts)], dim=0)

    @torch.no_grad()
    def extract_features(category, split):
        ds = MVTecSplit(args.data, category, split, preprocess)
        ld = DataLoader(ds, batch_size=args.batch_size, num_workers=2,
                        pin_memory=True, shuffle=False)
        feats, labels = [], []
        for imgs, lbls in ld:
            f = F.normalize(model.encode_image(imgs.to(device)), dim=-1)
            feats.append(f.cpu().numpy())
            labels.append(lbls.numpy())
        return np.concatenate(feats), np.concatenate(labels)

    @torch.no_grad()
    def windowed_scoremap(pil_img, text_embeds, grid=PIXEL_GRID):
        W, H = pil_img.size
        ww, wh = max(1, int(W * 0.40)), max(1, int(H * 0.40))
        xs = np.linspace(0, max(0, W - ww), grid).astype(int)
        ys = np.linspace(0, max(0, H - wh), grid).astype(int)
        crops = [preprocess(pil_img.crop((x, y, x + ww, y + wh)))
                 for y in ys for x in xs]
        batch = torch.stack(crops).to(device)
        feats = F.normalize(model.encode_image(batch), dim=-1)
        sim   = feats @ text_embeds.T * model.logit_scale.exp()
        p_ab  = sim.softmax(dim=-1)[:, 1].cpu().numpy().reshape(grid, grid)
        return np.asarray(
            Image.fromarray((p_ab * 255).astype(np.uint8))
                 .resize((PIXEL_RES, PIXEL_RES), Image.BICUBIC)
        ) / 255.0

    # ── Stage 1+3: image-level AUROC + prompt ablation ─────────────
    results = {}                       # category -> dict of aurocs
    abl = {"generic": [], "category": [], "ensemble": []}
    t0 = time.time()

    for cat in CATEGORIES:
        if not os.path.exists(os.path.join(args.data, cat)):
            print(f"SKIP {cat}")
            continue
        print(f"\n[{cat}]")
        obj = DISPLAY.get(cat, cat)

        test_feats, test_labels = extract_features(cat, "test")
        train_feats, _          = extract_features(cat, "train")
        has_both = len(np.unique(test_labels)) > 1

        # Method A: One-Class SVM (trained on normal images)
        scaler = StandardScaler()
        ocsvm  = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
        ocsvm.fit(scaler.fit_transform(train_feats))
        ocsvm_scores = -ocsvm.decision_function(scaler.transform(test_feats))
        ocsvm_auroc  = roc_auc_score(test_labels, ocsvm_scores) if has_both else float("nan")

        # Method B: CLIP Zero-Shot (full ensemble) + ablation variants
        def zs_auroc(text_embeds):
            sim   = test_feats @ text_embeds.cpu().numpy().T
            probs = torch.tensor(sim * model.logit_scale.exp().item()).softmax(dim=-1).numpy()
            return roc_auc_score(test_labels, probs[:, 1]) if has_both else float("nan")

        a_generic  = zs_auroc(text_pair(["a photo of a normal object"],
                                        ["a photo of a damaged object"]))
        a_category = zs_auroc(text_pair([f"a photo of a {obj}"],
                                        [f"a photo of a damaged {obj}"]))
        a_ensemble = zs_auroc(text_pair(build_prompts(cat, NORMAL_STATES),
                                        build_prompts(cat, ABNORMAL_STATES)))

        results[cat] = {"ocsvm": ocsvm_auroc, "zero_shot": a_ensemble}
        if has_both:
            abl["generic"].append(a_generic)
            abl["category"].append(a_category)
            abl["ensemble"].append(a_ensemble)

        win = "CLIP" if a_ensemble >= ocsvm_auroc else "OC-SVM"
        print(f"  OC-SVM (trained) : {ocsvm_auroc:.1%}")
        print(f"  CLIP zero-shot   : {a_ensemble:.1%}   <- {win}")

    # ── Stage 2: pixel-level AUROC (slow) ──────────────────────────
    pixel_aurocs = {}
    if not args.skip_pixel:
        print(f"\n{'='*58}\nPixel-level AUROC (localization) — this is the slow part\n{'='*58}")
        for cat in CATEGORIES:
            if not os.path.exists(os.path.join(args.data, cat)):
                continue
            te = text_pair(build_prompts(cat, NORMAL_STATES),
                           build_prompts(cat, ABNORMAL_STATES))
            scores_all, labels_all = [], []
            items = list_test_with_masks(args.data, cat)
            for img_path, mask_path in tqdm(items, desc=cat, leave=False):
                pil = Image.open(img_path).convert("RGB")
                smap = windowed_scoremap(pil, te)
                if mask_path is None:
                    mlab = np.zeros((PIXEL_RES, PIXEL_RES), dtype=np.uint8)
                else:
                    m = Image.open(mask_path).convert("L").resize(
                        (PIXEL_RES, PIXEL_RES), Image.NEAREST)
                    mlab = (np.asarray(m) > 0).astype(np.uint8)
                scores_all.append(smap.ravel())
                labels_all.append(mlab.ravel())
            s = np.concatenate(scores_all)
            l = np.concatenate(labels_all)
            pixel_aurocs[cat] = roc_auc_score(l, s) if len(np.unique(l)) > 1 else float("nan")
            print(f"  {cat:<15} pixel-AUROC: {pixel_aurocs[cat]:.1%}")

    # ── Summary ────────────────────────────────────────────────────
    def mean(d): return float(np.nanmean(list(d))) if len(d) else float("nan")

    mean_ocsvm = np.mean([v["ocsvm"]     for v in results.values()])
    mean_zs    = np.mean([v["zero_shot"] for v in results.values()])

    print(f"\n{'='*64}")
    print(f"{'Category':<14}{'OC-SVM':>10}{'CLIP-ZS':>10}{'Pixel-AUROC':>14}")
    print(f"{'-'*64}")
    for cat, r in results.items():
        px = pixel_aurocs.get(cat, float("nan"))
        mark = " *" if r["zero_shot"] >= r["ocsvm"] else ""
        print(f"{cat:<14}{r['ocsvm']:>9.1%}{r['zero_shot']:>10.1%}"
              f"{px:>13.1%}{mark}")
    print(f"{'-'*64}")
    print(f"{'MEAN':<14}{mean_ocsvm:>9.1%}{mean_zs:>10.1%}"
          f"{mean(pixel_aurocs.values()):>13.1%}")
    print(f"{'='*64}")

    print(f"\nPrompt ablation (mean image-level AUROC):")
    print(f"  Generic single prompt   : {mean(abl['generic']):.1%}")
    print(f"  Category single prompt  : {mean(abl['category']):.1%}")
    print(f"  Category + ensemble     : {mean(abl['ensemble']):.1%}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print(f"\n===== COPY INTO README =====")
    print(f"| One-Class SVM (trained) | {mean_ocsvm:.1%} | - |")
    print(f"| CLIP Zero-Shot (ours)   | {mean_zs:.1%} | {mean(pixel_aurocs.values()):.1%} |")
    print(f"Ablation: generic {mean(abl['generic']):.1%} -> "
          f"category {mean(abl['category']):.1%} -> "
          f"ensemble {mean(abl['ensemble']):.1%}")


if __name__ == "__main__":
    main()
