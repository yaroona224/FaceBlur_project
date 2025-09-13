

import os, sys, time, threading, atexit, signal, io
from pathlib import Path
import cv2, numpy as np
from flask import Flask, Response, render_template_string, request, send_file

# =========================
# CONFIG
# =========================
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "best.pt")

# Live stream settings (used by live frame processor)
IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))
CONF = float(os.environ.get("CONF", 0.28))
IOU  = float(os.environ.get("IOU", 0.5))

# Upload-only settings (high recall, conservative merges)
UPLOAD_CONF   = float(os.environ.get("UPLOAD_CONF", 0.15))  # lower to catch tiny text
UPLOAD_IOU    = float(os.environ.get("UPLOAD_IOU", 0.5))
TILE_SIZE     = int(os.environ.get("TILE_SIZE", 960))
TILE_OVERLAP  = float(os.environ.get("TILE_OVERLAP", 0.20))

# Text post-process
TEXT_DILATE_FRAC    = float(os.environ.get("TEXT_DILATE_FRAC", 0.010))   # small dilation each side
TEXT_MERGE_GAP_FRAC = float(os.environ.get("TEXT_MERGE_GAP_FRAC", 0.010)) # ≤1.0% gap to merge
TEXT_MAX_DOC_FRAC   = float(os.environ.get("TEXT_MAX_DOC_FRAC", 0.50))    # cap merged text area vs doc
TEXT_MIN_H_FRAC     = float(os.environ.get("TEXT_MIN_H_FRAC", 0.012))     # ignore boxes <1.2% of H
TEXT_MAX_H_FRAC     = float(os.environ.get("TEXT_MAX_H_FRAC", 0.28))      # ignore boxes >28% of H
TEXT_MIN_AR         = float(os.environ.get("TEXT_MIN_AR", 2.3))           # min aspect ratio (w/h)
TEXT_MAX_AR         = float(os.environ.get("TEXT_MAX_AR", 40.0))          # max aspect ratio
TEXT_NMS_IOU        = float(os.environ.get("TEXT_NMS_IOU", 0.35))         # de-dup text

# OCR tuning (optional, multi-language)
USE_OCR             = os.environ.get("USE_OCR", "1") != "0"
OCR_LANG            = os.environ.get("OCR_LANG", "en")       # "ar" or "en" or "ar,en"
OCR_MIN_CONF        = float(os.environ.get("OCR_MIN_CONF", 0.60))
OCR_EXPAND_PX       = int(os.environ.get("OCR_EXPAND_PX", 2))

# Blur strength
MIN_KERNEL   = int(os.environ.get("MIN_KERNEL", 31))
KERNEL_SCALE = float(os.environ.get("KERNEL_SCALE", 0.22))

# Camera preferences (kept but NOT used for server-side webcam anymore)
DEFAULT_CAM_INDEX = int(os.environ.get("CAM_INDEX", 0))
FORCE_DEVICE = os.environ.get("FORCE_DEVICE", "").strip().lower()  # "", "cpu", "cuda", "mps"

# Labels (match your model)
DOC_LABELS  = {"id", "id_card", "idcard", "passport", "mrz", "serial", "number", "document", "card", "passport_id", "name", "dob", "expiry"}
FACE_LABELS = {"face", "person_face", "head"}
TEXT_LABELS = {"text"}

# Anti-flicker (for doc blurring)
DOC_PAD_FRAC = float(os.environ.get("DOC_PAD_FRAC", 0.08))

# ============== helpers ==============
def _lazy_import():
    from ultralytics import YOLO as _YOLO
    import torch as _torch
    return _YOLO, _torch

def pick_device(_torch):
    if FORCE_DEVICE in ("cpu", "cuda", "mps"): return FORCE_DEVICE
    if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available(): return "mps"
    if _torch.cuda.is_available(): return "cuda"
    return "cpu"

def make_odd(n): return n if n % 2 == 1 else n + 1
def compute_kernel(w, h):
    k = int(max(w, h) * KERNEL_SCALE)
    k = max(k, MIN_KERNEL)
    return make_odd(k)

def blur_region(img, x1, y1, x2, y2):
    H, W = img.shape[:2]
    x1 = max(0, min(W - 1, x1)); y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2));     y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1: return
    k = compute_kernel(x2 - x1, y2 - y1)
    if k < 3: k = 3
    img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (k, k), 0)

def center(x1, y1, x2, y2): return ((x1 + x2) // 2, (y1 + y2) // 2)
def contains(box, pt):
    x1, y1, x2, y2 = box; x, y = pt
    return (x1 <= x <= x2) and (y1 <= y <= y2)

def pad_box(box, pad_frac, W, H):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad_frac), int(h * pad_frac)
    nx1 = max(0, x1 - px); ny1 = max(0, y1 - py)
    nx2 = min(W, x2 + px); ny2 = min(H, y2 + py)
    return (nx1, ny1, nx2, ny2)

def expand_px(box, px=4, py=4, W=99999, H=99999):
    x1, y1, x2, y2 = box
    return (max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py))

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ub = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (ua + ub - inter + 1e-6)

def box_area(b):
    x1,y1,x2,y2=b
    return max(0,x2-x1)*max(0,y2-y1)

def merge_boxes_overlap_or_near(boxes, max_gap_px):
    """Conservative merge: only overlap or very small gap with orthogonal overlap."""
    if not boxes: return []
    boxes = boxes[:]
    merged = []
    used = [False]*len(boxes)

    def near_or_overlap(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        if not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1):
            return True
        h_gap = max(0, max(bx1 - ax2, ax1 - bx2))
        v_overlap = min(ay2, by2) - max(ay1, by1)
        if h_gap <= max_gap_px and v_overlap > 0: return True
        v_gap = max(0, max(by1 - ay2, ay1 - by2))
        h_overlap = min(ax2, bx2) - max(ax1, bx1)
        if v_gap <= max_gap_px and h_overlap > 0: return True
        return False

    for i in range(len(boxes)):
        if used[i]: continue
        cur = boxes[i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]: continue
                if near_or_overlap(cur, boxes[j]):
                    x1 = min(cur[0], boxes[j][0]); y1 = min(cur[1], boxes[j][1])
                    x2 = max(cur[2], boxes[j][2]); y2 = max(cur[3], boxes[j][3])
                    cur = (x1,y1,x2,y2)
                    used[j] = True
                    changed = True
        merged.append(cur)
    return merged

# ---- NMS helper (used for faces & text de-dup) ----
def nms_boxes(boxes, scores=None, iou_th=0.5):
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=float)
    if scores is None:
        scores = np.ones(len(boxes), dtype=float)
    else:
        scores = np.array(scores, dtype=float)
    order = scores.argsort()[::-1]
    keep = []
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0: return 0.0
        ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        ub = max(0, bx2 - bx1) * max(0, by2 - by1)
        return inter / (ua + ub - inter + 1e-6)
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        suppress = []
        for j in rest:
            if _iou(boxes[i], boxes[int(j)]) > iou_th:
                suppress.append(j)
        order = np.array([k for k in rest if k not in suppress])
    return keep

# =========================
# Upload processor — YOLO + multi-lang OCR fusion + conservative text merge
# =========================
class ImageProcessor:
    def __init__(self):
        if not WEIGHTS_PATH or not Path(WEIGHTS_PATH).exists():
            raise FileNotFoundError(f"WEIGHTS_PATH not found: {WEIGHTS_PATH}")
        YOLO, torch = _lazy_import()
        self.device = pick_device(torch)
        self.model = YOLO(WEIGHTS_PATH)
        try: self.model.fuse()
        except Exception: pass
        self.names = getattr(self.model, "names", {})

        # Optional multi-language OCR (version-safe init: no show_log)
        self.ocrs = []
        if USE_OCR:
            langs = [s.strip() for s in OCR_LANG.split(",") if s.strip()]
            try:
                from paddleocr import PaddleOCR
                for lang in langs:
                    ocr = None
                    try:
                        # Newer PaddleOCR
                        ocr = PaddleOCR(lang=lang, use_textline_orientation=True)
                    except TypeError:
                        # Older PaddleOCR
                        ocr = PaddleOCR(lang=lang, use_angle_cls=True)
                    except Exception as e_any:
                        print(f"[WARN] OCR init failed for lang='{lang}':", e_any)
                    if ocr is not None:
                        self.ocrs.append(ocr)
                        print(f"[INFO] PaddleOCR initialized for lang='{lang}'.")
                if not self.ocrs:
                    print("[WARN] No OCR engines initialized; OCR disabled.")
            except Exception as e:
                print("[WARN] OCR disabled (import/init failed):", e)

    def _predict(self, img_rgb, conf, iou):
        try:
            results = self.model.predict(img_rgb, imgsz=IMG_SIZE, conf=conf, iou=iou, device=self.device, verbose=False)
        except Exception as e:
            print("[WARN] Upload inference error:", e, file=sys.stderr)
            return []
        out = []
        if not results: return out
        res = results[0]
        def npint(x): return x.cpu().numpy().astype(int)
        def npfloat(x): return x.cpu().numpy().astype(float)
        try:
            if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                xyxy = npint(res.boxes.xyxy)
                cls  = res.boxes.cls.int().tolist() if hasattr(res.boxes, "cls") else [None]*len(xyxy)
                scr  = npfloat(res.boxes.conf).tolist() if hasattr(res.boxes, "conf") else [1.0]*len(xyxy)
                for coords, c, s in zip(xyxy.tolist(), cls, scr):
                    label = self.names.get(int(c), str(c))
                    out.append((tuple(coords), (label or "").lower(), float(s)))
        except Exception: pass
        try:
            if hasattr(res, "obb") and res.obb is not None and len(res.obb) > 0:
                xyxy = npint(res.obb.xyxy)
                cls  = res.obb.cls.int().tolist() if hasattr(res.obb, "cls") else [None]*len(xyxy)
                scr  = npfloat(res.obb.conf).tolist() if hasattr(res.obb, "conf") else [1.0]*len(xyxy)
                for coords, c, s in zip(xyxy.tolist(), cls, scr):
                    label = self.names.get(int(c), str(c))
                    out.append((tuple(coords), (label or "").lower(), float(s)))
        except Exception: pass
        return out

    def _predict_tiled(self, img_rgb, conf, iou):
        H, W = img_rgb.shape[:2]
        ts = min(TILE_SIZE, max(H, W))
        step_x = int(ts * (1 - TILE_OVERLAP))
        step_y = int(ts * (1 - TILE_OVERLAP))
        if step_x <= 0 or step_y <= 0:
            return self._predict(img_rgb, conf, iou)
        out = []
        for y in range(0, H, step_y):
            for x in range(0, W, step_x):
                x2 = min(x + ts, W)
                y2 = min(y + ts, H)
                tile = img_rgb[y:y2, x:x2]
                preds = self._predict(tile, conf, iou)
                for (bx1, by1, bx2, by2), label, score in preds:
                    out.append(((bx1 + x, by1 + y, bx2 + x, by2 + y), label, score))
        return out

    def _predict_tta_flip(self, img_rgb, conf, iou):
        H, W = img_rgb.shape[:2]
        flip = cv2.flip(img_rgb, 1)
        preds = self._predict(flip, conf, iou)
        mapped = []
        for (x1, y1, x2, y2), label, score in preds:
            nx1 = W - x2
            nx2 = W - x1
            mapped.append(((nx1, y1, nx2, y2), label, score))
        return mapped

    def _gather_boxes(self, preds):
        doc_boxes, face_boxes, face_scores, text_boxes, text_scores = [], [], [], [], []
        for (x1,y1,x2,y2), label, score in preds:
            if label in DOC_LABELS:    doc_boxes.append((x1,y1,x2,y2))
            elif label in FACE_LABELS: face_boxes.append((x1,y1,x2,y2)); face_scores.append(score)
            elif label in TEXT_LABELS: text_boxes.append((x1,y1,x2,y2)); text_scores.append(score)
        if face_boxes:
            keep = nms_boxes(face_boxes, face_scores, iou_th=0.45)
            face_boxes = [face_boxes[i] for i in keep]
        if text_boxes:
            keep = nms_boxes(text_boxes, None, iou_th=TEXT_NMS_IOU)
            text_boxes = [text_boxes[i] for i in keep]
        return doc_boxes, face_boxes, text_boxes

    # ---------- OCR helpers ----------
    def _ocr_boxes_single(self, ocr_engine, img_rgb):
        try:
            # try modern signature
            result = ocr_engine.ocr(img_rgb, cls=True)
        except TypeError:
            result = ocr_engine.ocr(img_rgb)
        except Exception as e:
            print("[WARN] OCR error:", e)
            return []
        boxes = []
        if not result: return boxes
        for line in result[0]:
            quad, info = line
            # info could be (text, conf) or dict-like
            if isinstance(info, (list, tuple)) and len(info) >= 2:
                conf = float(info[1]) if info[1] is not None else 1.0
            elif isinstance(info, dict) and "score" in info:
                conf = float(info["score"])
            else:
                conf = 1.0
            if conf < OCR_MIN_CONF:
                continue
            xs = [int(p[0]) for p in quad]; ys = [int(p[1]) for p in quad]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            boxes.append((x1 - OCR_EXPAND_PX, y1 - OCR_EXPAND_PX, x2 + OCR_EXPAND_PX, y2 + OCR_EXPAND_PX))
        return boxes

    def _ocr_boxes_multi(self, img_rgb):
        if not self.ocrs: return []
        all_boxes = []
        for ocr in self.ocrs:
            all_boxes.extend(self._ocr_boxes_single(ocr, img_rgb))
        if all_boxes:
            keep = nms_boxes(all_boxes, None, iou_th=0.4)
            all_boxes = [all_boxes[i] for i in keep]
        return all_boxes

    def _filter_text_geometry(self, text_boxes, W, H):
        out = []
        for (x1,y1,x2,y2) in text_boxes:
            w = max(1, x2 - x1); h = max(1, y2 - y1)
            h_frac = h / H; ar = w / h
            if h_frac < TEXT_MIN_H_FRAC: continue
            if h_frac > TEXT_MAX_H_FRAC: continue
            if ar < TEXT_MIN_AR or ar > TEXT_MAX_AR:
                # keep small near-square if very short (digits/short words)
                if not (h_frac < (TEXT_MIN_H_FRAC*1.5) and 1.2 <= ar <= 2.5):
                    continue
            out.append((x1,y1,x2,y2))
        return out

    def _group_by_rows(self, boxes, row_tol_px):
        if not boxes: return []
        ys = [ ( (b[1]+b[3])//2, i ) for i,b in enumerate(boxes) ]
        ys.sort()
        groups, cur, last_y = [], [], None
        for yc, i in ys:
            if last_y is None or abs(yc - last_y) <= row_tol_px:
                cur.append(i)
            else:
                groups.append(cur); cur=[i]
            last_y = yc
        if cur: groups.append(cur)
        return [[boxes[i] for i in g] for g in groups]

    def _conservative_text(self, text_boxes, doc_boxes, W, H):
        if not text_boxes: return []
        # 1) geometry filter
        text_boxes = self._filter_text_geometry(text_boxes, W, H)
        if not text_boxes: return []

        # 2) tiny dilation
        px = max(2, int(TEXT_DILATE_FRAC * W))
        py = max(2, int(TEXT_DILATE_FRAC * H))
        dil = [expand_px(t, px=px, py=py, W=W, H=H) for t in text_boxes]

        # 3) group by rows to avoid vertical over-merge
        row_tol = max(3, int(0.012 * H))  # ~1.2% of height
        row_groups = self._group_by_rows(dil, row_tol_px=row_tol)

        gap = int(TEXT_MERGE_GAP_FRAC * max(W, H))
        merged_all = []
        for grp in row_groups:
            merged_all.extend(merge_boxes_overlap_or_near(grp, max_gap_px=gap))

        # 4) cap vs containing doc
        if doc_boxes and merged_all:
            safe = []
            for m in merged_all:
                cx, cy = center(*m)
                chosen = None
                for d in doc_boxes:
                    if contains(d, (cx, cy)):
                        chosen = d; break
                if chosen is None and doc_boxes:
                    ious = [iou(m, d) for d in doc_boxes]
                    chosen = doc_boxes[int(np.argmax(ious))] if any(ious) else None
                if chosen and box_area(m) > TEXT_MAX_DOC_FRAC * box_area(chosen):
                    inside = [b for b in dil if contains(chosen, center(*b))]
                    safe.extend(merge_boxes_overlap_or_near(inside, max_gap_px=gap))
                else:
                    safe.append(m)
            merged_all = safe

        # 5) final de-dup
        if merged_all:
            keep = nms_boxes(merged_all, None, iou_th=0.4)
            merged_all = [merged_all[i] for i in keep]

        return merged_all

    def run_upload(self, bgr_image: np.ndarray, mode: str = "both") -> np.ndarray:
        """Used for the Upload buttons (text/face/both/doc)."""
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        preds = []
        preds += self._predict(rgb, UPLOAD_CONF, UPLOAD_IOU)
        preds += self._predict_tiled(rgb, UPLOAD_CONF, UPLOAD_IOU)
        preds += self._predict_tta_flip(rgb, UPLOAD_CONF, UPLOAD_IOU)

        doc_boxes, face_boxes, text_boxes = self._gather_boxes(preds)

        # OCR fusion (Arabic, English, etc.)
        ocr_boxes = self._ocr_boxes_multi(rgb)
        if ocr_boxes:
            if doc_boxes:
                ocr_boxes = [b for b in ocr_boxes if any(contains(d, center(*b)) for d in doc_boxes)]
            combined = text_boxes + ocr_boxes
            keep = nms_boxes(combined, None, iou_th=0.4)
            text_boxes = [combined[i] for i in keep]

        H, W = bgr_image.shape[:2]

        # keep only text/face inside a detected doc (if any)
        if doc_boxes:
            text_boxes = [t for t in text_boxes if any(contains(d, center(*t)) for d in doc_boxes)]
            face_boxes = [f for f in face_boxes if any(contains(d, center(*f)) for d in doc_boxes)]

        # smarter text merge
        text_boxes = self._conservative_text(text_boxes, doc_boxes, W, H)

        m = (mode or "both").lower()
        if m == "text":
            targets = text_boxes
        elif m == "face":
            targets = face_boxes
        elif m == "both":
            merged = text_boxes + face_boxes
            targets = [merged[i] for i in nms_boxes(merged, None, iou_th=0.3)] if merged else []
        elif m == "doc":
            targets = [pad_box(d, DOC_PAD_FRAC, W, H) for d in doc_boxes]
        else:
            targets = []

        for (x1, y1, x2, y2) in targets:
            blur_region(bgr_image, x1, y1, x2, y2)
        return bgr_image

    def run_live_doc_only(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Live stream behavior like your original: blur WHOLE DOC only when a document
        exists AND it contains both a face and text. (Anti-flicker handled by client fps)
        """
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        preds = self._predict(rgb, CONF, IOU)

        # gather
        doc_boxes, face_boxes, text_boxes = [], [], []
        for (x1,y1,x2,y2), label, _score in preds:
            if label in DOC_LABELS:    doc_boxes.append((x1,y1,x2,y2))
            elif label in FACE_LABELS: face_boxes.append((x1,y1,x2,y2))
            elif label in TEXT_LABELS: text_boxes.append((x1,y1,x2,y2))

        H, W = bgr_image.shape[:2]
        blur_regions = []
        if doc_boxes:
            for d in doc_boxes:
                faces_in = [f for f in face_boxes if contains(d, center(*f))]
                texts_in = [t for t in text_boxes if contains(d, center(*t))]
                if faces_in and texts_in:
                    d_padded = pad_box(d, DOC_PAD_FRAC, W, H)
                    blur_regions.append(d_padded)

        for (x1, y1, x2, y2) in merge_boxes_overlap_or_near(blur_regions, max_gap_px=0):
            blur_region(bgr_image, x1, y1, x2, y2)
        return bgr_image

# =========================
# Flask app (layout unchanged; live uses visitor webcam)
# =========================
app = Flask(__name__)
uploader = ImageProcessor()
_upload_jpeg = None

INDEX_HTML = """
<!doctype html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>FaceBlur</title>
  <style>
    :root{
      --bg:#0f1115; --card:#151923; --muted:#99a1b3; --text:#e8ecf3;
      --border:#202637; --accent:#9c80ff; --shadow:rgba(0,0,0,.20);
      --accent-2:#7c3aed;
    }
    [data-theme="light"]{
      --bg:#f7f8fb; --card:#ffffff; --muted:#667085; --text:#0f141e;
      --border:#e6e8ee; --accent:#6d28d9; --shadow:rgba(0,0,0,.06);
      --accent-2:#5b21b6;
    }
    html,body{height:100%}
    body{ margin:0; background:var(--bg); color:var(--text);
      font-family: ui-sans-serif,system-ui,-apple-system,Segoe UI,Inter,Roboto,Helvetica,Arial,sans-serif; }
    .container{max-width:1200px;margin:0 auto;padding:28px 18px}
    .header{display:flex; align-items:center; justify-content:space-between; margin-bottom:10px}
    .brand{display:flex; align-items:center; gap:12px}
    .brand h1{font-size:46px; margin:0} /* slightly bigger title */
    .tagline{font-size:14px; color:var(--muted); margin:4px 0 0 2px}
    .toggle{width:42px;height:24px;border-radius:999px;background:var(--card);border:1px solid var(--border);position:relative;cursor:pointer}
    .toggle::after{content:"";position:absolute;top:2px;left:2px;width:18px;height:18px;border-radius:50%;background:var(--text);transition:.15s ease}
    [data-theme="light"] .toggle::after{ left:22px }
    .row{display:flex; gap:10px; align-items:center; flex-wrap:wrap}
    .btn{
      background:var(--card); color:var(--text); border:1px solid var(--border);
      padding:10px 14px; border-radius:12px; cursor:pointer; transition:.15s ease; font-weight:600;
    }
    .btn[disabled]{opacity:.5; cursor:not-allowed}
    .btn:hover{border-color:var(--accent)}
    .btn-primary{
      background: linear-gradient(180deg, var(--accent), var(--accent-2));
      color:white; border:none; box-shadow:0 10px 24px var(--shadow);
    }
    .btn-primary:hover{ filter:brightness(1.05) }
    .btn-lg{ font-size:16px; padding:14px 18px; border-radius:14px; }
    .card{background:var(--card); border:1px solid var(--border); border-radius:16px; padding:14px; box-shadow:0 10px 30px var(--shadow)}
    .video{overflow:hidden; border-radius:14px; background:#000}
    .video img{display:block; width:100%; height:auto}
    .status{font-size:12px; color:var(--muted); margin:10px 2px 12px}
    .footer{margin-top:18px; font-size:14px; color:var(--text); font-weight:700; text-align:center}
    .sep{opacity:.35; margin:0 8px}
    .upload-wrap{display:grid; grid-template-columns: 1fr 560px; gap:16px; margin-top:16px}
    @media (max-width:1200px){ .upload-wrap{grid-template-columns:1fr} }
    .preview{overflow:hidden; border-radius:12px; background:#0a0a0a; border:1px solid var(--border); min-height:320px; display:flex; align-items:center; justify-content:center}
    .preview img{max-width:100%; height:auto; display:block; cursor:zoom-in}
    input[type=file]{display:block; width:100%; padding:10px; border-radius:10px; border:1px solid var(--border); background:var(--card); color:var(--text)}
    .options{display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:8px}
    a.small{font-size:12px; color:var(--muted); text-decoration:none}
    /* Make the Download button stand out */
    #downloadBtn{ grid-column: span 2; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <div class="brand">
          <svg width="38" height="38" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M4 7a3 3 0 0 1 3-3h4l2 2h4a3 3 0 0 1 3 3v8a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3V7Z"
                  stroke="currentColor" stroke-width="1.5"/>
            <path d="M8.5 14.5c2.5-2.5 4.5-2.5 7 0" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
          <h1>FaceBlur</h1>
        </div>
        <p class="tagline">AI-powered privacy protection for IDs and passports</p>
      </div>
      <div class="toggle" id="themeToggle" title="Toggle theme" aria-label="Toggle theme"></div>
    </div>

    <!-- Camera controls -->
    <div class="row" style="margin-bottom:8px;">
      <button class="btn" id="startCam">Start Camera</button>
      <button class="btn" id="stopCam">Turn Off</button>
    </div>
    <div class="status" id="statusText">Camera is OFF</div>

    <!-- Live video -->
    <div class="card video">
      <img id="stream" src="" alt="video stream"/>
    </div>

    <!-- Upload & process (bigger preview) -->
    <div class="upload-wrap">
      <div class="card">
        <h3 style="margin:6px 0 10px 0;">Process an Image</h3>
        <input type="file" id="fileInput" accept="image/*" aria-label="Choose image to blur"/>
        <div class="options" style="margin-top:10px;">
          <button class="btn" data-mode="text">Blur TEXT only</button>
          <button class="btn" data-mode="face">Blur FACE only</button>
          <button class="btn" data-mode="both">Blur BOTH</button>
          <button class="btn" data-mode="doc">Blur WHOLE CARD</button>
          <!-- Bigger, primary download button -->
          <button class="btn btn-primary btn-lg" id="downloadBtn" disabled aria-disabled="true" aria-label="Download blurred image">⬇ Download Blurred Image</button>
        </div>
        <p class="status" style="margin-top:10px;">Choose an image, then select a blur option. Click download only if you want to save it.</p>
        <p><a class="small" href="/upload_result" target="_blank">Open processed image in new tab</a></p>
      </div>
      <div class="preview card" aria-live="polite">
        <img id="preview" src="" alt="preview will appear here" title="Click to open full size"/>
      </div>
    </div>

    <!-- Footer -->
    <div class="footer">
      Shatha Khawaji <span class="sep">•</span> Renad Almutairi <span class="sep">•</span> Jury Alsultan <span class="sep">•</span> Yara Alsardi
      <div style="font-size:12px; color:var(--muted); margin-top:6px;">Built as our AI capstone project</div>
    </div>
  </div>

  <script>
    // Theme toggle
    const root = document.documentElement;
    const saved = localStorage.getItem('theme') || 'dark';
    root.setAttribute('data-theme', saved);
    document.getElementById('themeToggle').addEventListener('click', ()=>{
      const next = (root.getAttribute('data-theme') === 'dark') ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
    });

    const streamImg  = document.getElementById('stream');
    const statusText = document.getElementById('statusText');

    // ======= Live camera via visitor's browser (HTTPS required on public URL) =======
    let camStream = null;
    let running   = false;
    let rafId     = null;
    let lastSent  = 0;

    const off = document.createElement('canvas');
    const ctx = off.getContext('2d');

    async function startClientCamera(){
      // Camera permission: allowed on https:// or localhost
      camStream = await navigator.mediaDevices.getUserMedia({ video:true, audio:false });
      running = true;
      statusText.textContent = 'Camera active — blurring sensitive info in real-time';
      pushFrames();
    }

    function stopClientCamera(){
      running = false;
      if (rafId) cancelAnimationFrame(rafId);
      rafId = null;
      if (camStream){
        camStream.getTracks().forEach(t=>t.stop());
        camStream = null;
      }
      streamImg.src = '';
      statusText.textContent = 'Camera is OFF';
    }

    async function pushFrames(){
      const fps = 4;
      const minInterval = 1000 / fps;

      const videoEl = document.createElement('video');
      videoEl.srcObject = camStream;
      videoEl.muted = true;
      await videoEl.play();

      const loop = async ()=>{
        if (!running || !camStream){ return; }
        const now = performance.now();
        if (now - lastSent >= minInterval && videoEl.videoWidth){
          lastSent = now;
          const maxW = 720;
          const scale = Math.min(1, maxW / videoEl.videoWidth);
          off.width  = Math.round(videoEl.videoWidth  * scale);
          off.height = Math.round(videoEl.videoHeight * scale);
          ctx.drawImage(videoEl, 0, 0, off.width, off.height);
          const blob = await new Promise(res=>off.toBlob(res, 'image/jpeg', 0.8));
          const form = new FormData();
          form.append('frame', blob, 'frame.jpg');
          try{
            const r = await fetch('/process_frame', { method:'POST', body: form });
            if (r.ok){
              const arr = await r.arrayBuffer();
              const url = URL.createObjectURL(new Blob([arr], {type:'image/jpeg'}));
              streamImg.src = url;
            }
          }catch(_e){}
        }
        rafId = requestAnimationFrame(loop);
      };
      rafId = requestAnimationFrame(loop);
    }

    // Buttons (keep same UI)
    document.getElementById('startCam').onclick = async ()=>{
      try{
        await startClientCamera();
      }catch(e){
        alert('Could not access camera. On public links, use HTTPS (e.g., via ngrok).');
      }
    };

    document.getElementById('stopCam').onclick = ()=> stopClientCamera();

    // Hidden shortcut: 'r' to "refresh" current frame
    window.addEventListener('keydown', (e)=>{
      if (e.key.toLowerCase() === 'r' && streamImg.src)
        streamImg.src = streamImg.src; // noop refresh
    });

    // ======= Upload + modes (unchanged) =======
    const fileInput   = document.getElementById('fileInput');
    const preview     = document.getElementById('preview');
    const downloadBtn = document.getElementById('downloadBtn');

    function enableDownload(){ downloadBtn.disabled = false; downloadBtn.setAttribute('aria-disabled','false'); }
    function disableDownload(){ downloadBtn.disabled = true;  downloadBtn.setAttribute('aria-disabled','true'); }

    document.querySelectorAll('.options .btn').forEach(btn=>{
      if (btn.id === 'downloadBtn') return; // handled separately
      btn.addEventListener('click', async ()=>{
        const mode = btn.getAttribute('data-mode');
        const file = fileInput.files && fileInput.files[0];
        if (!file){ alert('Please choose an image first.'); return; }
        const form = new FormData();
        form.append('image', file);
        form.append('mode', mode);
        disableDownload();
        const res = await fetch('/upload', {method:'POST', body: form});
        if (res.ok){
          const url = '/upload_result?ts=' + Date.now();
          preview.src = url;
          enableDownload();
        } else {
          alert('Processing failed.');
        }
      });
    });

    downloadBtn.addEventListener('click', ()=>{
      if (!downloadBtn.disabled){
        window.location.href = '/download_result?ts=' + Date.now();
      }
    });

    preview.addEventListener('click', ()=>{
      if (preview.src) window.open('/upload_result?ts=' + Date.now(), '_blank');
    });
  </script>
</body>
</html>
"""

# -------- Camera start/stop (kept for compatibility; no longer starts server webcam) --------
@app.route("/cam/start", methods=["POST"])
def cam_start():
    # We now use the visitor's browser camera; nothing to start server-side.
    return ("OK", 200)

@app.route("/cam/stop", methods=["POST"])
def cam_stop():
    return ("OK", 200)

# -------- Base pages --------
@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

# -------- Live frame endpoint (visitor webcam → server blur → jpeg back) --------
@app.route("/process_frame", methods=["POST"])
def process_frame():
    if "frame" not in request.files:
        return ("No frame", 400)
    data = np.frombuffer(request.files["frame"].read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return ("Bad frame", 400)
    out = uploader.run_live_doc_only(bgr)
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return ("Encode error", 500)
    return Response(buf.tobytes(), mimetype="image/jpeg")

# -------- Upload routes (unchanged UI) --------
_upload_jpeg = None

@app.route("/upload", methods=["POST"])
def upload():
    global _upload_jpeg
    if "image" not in request.files:
        return ("No file", 400)
    mode = request.form.get("mode", "both")
    data = np.frombuffer(request.files["image"].read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return ("Bad image", 400)
    out = uploader.run_upload(bgr, mode=mode)
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        return ("Encode error", 500)
    _upload_jpeg = buf.tobytes()
    return ("OK", 200)

@app.route("/upload_result")
def upload_result():
    global _upload_jpeg
    if not _upload_jpeg:
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", blank, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        _upload_jpeg = buf.tobytes() if ok else b""
    return send_file(io.BytesIO(_upload_jpeg), mimetype="image/jpeg", as_attachment=False, download_name="result.jpg")

@app.route("/download_result")
def download_result():
    global _upload_jpeg
    if not _upload_jpeg:
        return ("No processed image available to download. Process one first.", 400)
    return send_file(io.BytesIO(_upload_jpeg), mimetype="image/jpeg", as_attachment=True, download_name="faceblur_result.jpg")

# ---- lifecycle cleanup ----
def _cleanup(*_):
    pass

atexit.register(_cleanup)
signal.signal(signal.SIGINT,  _cleanup)
signal.signal(signal.SIGTERM, _cleanup)

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5000))
    print(f"[INFO] Visit http://127.0.0.1:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True, use_reloader=False)
