"""
Zero-Shot Industrial Anomaly Detection — CLIP (Vision-Language Model)
====================================================================
Upload a product image, pick its category — the model flags it as
Normal or Anomalous with ZERO task-specific
training.

Method:
  • Image-level score: CLIP whole-image embedding vs. ensembled
    "normal"/"abnormal" text prompts (WinCLIP-style prompt ensembling).
  • Localization: the image is scored over a grid of overlapping
    windows; per-window anomaly scores form a heatmap (the WinCLIP
    multi-window idea, simplified).
"""

import gradio as gr
import torch
import torch.nn.functional as F

# ── Prompt ensembles ──────────────────────────────────────────────
CATEGORY_DISPLAY_NAMES = {
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


def _build_prompts(category: str, states):
    obj = CATEGORY_DISPLAY_NAMES.get(category, category)
    return [t.format(state=s, object=obj) for t in TEMPLATES for s in states]


# ── Model (lazy load, ViT-B/32 for CPU-friendly free HF Spaces) ───
_DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
_model      = None
_preprocess = None
_tokenizer  = None


def _load():
    global _model, _preprocess, _tokenizer
    if _model is None:
        import open_clip
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        _model = _model.to(_DEVICE).eval()
        _tokenizer = open_clip.get_tokenizer("ViT-B-32")


@torch.no_grad()
def _encode(prompts):
    tokens = _tokenizer(prompts).to(_DEVICE)
    feats  = F.normalize(_model.encode_text(tokens), dim=-1)
    return F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)



@torch.no_grad()
    crops = []
    for y in ys:
        for x in xs:
            crop = pil_img.crop((x, y, x + win_w, y + win_h))
            crops.append(_preprocess(crop))
    batch = torch.stack(crops).to(_DEVICE)                       # (grid*grid, C, H, W)

    feats = F.normalize(_model.encode_image(batch), dim=-1)
    sim   = feats @ text_embeds.T * _model.logit_scale.exp()
    p_ab  = sim.softmax(dim=-1)[:, 1].cpu().numpy()              # (grid*grid,)
    score_grid = p_ab.reshape(grid, grid)

    # Normalize for visualization, upsample to full image, colorize, blend
    g = score_grid - score_grid.min()
    g = g / (g.max() + 1e-8)
    heat = Image.fromarray((g * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    heat_rgb = _colormap(np.asarray(heat) / 255.0)
    base = np.asarray(pil_img.convert("RGB")).astype(np.float32)
    blended = (0.55 * base + 0.45 * heat_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ── Inference ─────────────────────────────────────────────────────
@torch.no_grad()
def detect(image, category):
    if image is None:
        return {"Upload an image": 1.0}, ""

    _load()
    pil = image.convert("RGB")

    # Image-level score (headline verdict)
    img_t    = _preprocess(pil).unsqueeze(0).to(_DEVICE)
    img_feat = F.normalize(_model.encode_image(img_t), dim=-1)
    text_embeds = torch.cat([
        _encode(_build_prompts(category, NORMAL_STATES)),
        _encode(_build_prompts(category, ABNORMAL_STATES)),
    ], dim=0)
    probs      = (img_feat @ text_embeds.T * _model.logit_scale.exp()).softmax(dim=-1).squeeze()
    p_abnormal = probs[1].item()
    verdict    = "ANOMALOUS" if p_abnormal >= 0.5 else "NORMAL"

    # Localization heatmap

    label = {"Anomalous": round(p_abnormal, 4), "Normal": round(1 - p_abnormal, 4)}
    info  = (
        f"**Verdict: {verdict}** &nbsp;|&nbsp; anomaly score = `{p_abnormal:.3f}`\n\n"
        f"Category: **{CATEGORY_DISPLAY_NAMES.get(category, category)}** · "
        f"Model: CLIP ViT-B/32 · No training data used.\n\n"
    )
    return label, info


# ── UI ────────────────────────────────────────────────────────────
with gr.Blocks(title="Zero-Shot Anomaly Detection") as demo:
    gr.Markdown(
        "## Zero-Shot Industrial Anomaly Detection with CLIP\n"
        "Upload a product image and select its category. The model detects **and "
        "defects **without any task-specific training**, using a "
        "vision-language model (CLIP)."
    )
    with gr.Row():
        with gr.Column(scale=1):
            inp_image = gr.Image(type="pil", label="Product image")
            inp_cat   = gr.Dropdown(
                choices=sorted(CATEGORY_DISPLAY_NAMES),
                value="bottle",
                label="Product category (MVTec AD)",
            )
            btn = gr.Button("Detect", variant="primary")
        with gr.Column(scale=1):
            out_label   = gr.Label(num_top_classes=2, label="Prediction")
            out_info    = gr.Markdown()

    btn.click(detect, inputs=[inp_image, inp_cat],

if __name__ == "__main__":
    demo.launch()
