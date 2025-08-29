# src/inference.py — LARGE-SCALE INFERENCE (AUC AML mapping + robust thresholding + optional temp scaling)

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import tifffile
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)

# =========================
# Dataset (balanced + large)
# =========================
class BalancedTestDataset(Dataset):
    """
    Balanced dataset using *percentage ranges* of the full directory listing:

      "cal"  : 60–70%  (calibration)
      "test" : 70–100% (evaluation)

    Returns both:
      - 'label' (int: 1=AML, 0=Healthy) for easy batching, and
      - 'labels' dict (for compatibility with older code), and
      - 'path' for auditability.
    """

    def __init__(
        self,
        root="/scratch/project_2010376/BDS8/BDS8_data",
        test_per_class=10000,
        seed=42,
        region="test",
        exclude_paths=None,
        img_size=512,  # tune in config if needed
    ):
        rng = random.Random(seed)
        self.img_size = int(img_size)
        self.images, self.label_vec, self.labels_dict, self.paths = [], [], [], []

        aml_dir = Path(root) / "AML"
        hea_dir = Path(root) / "Healthy BM"

        # Both .tif and .tiff
        aml_all = sorted(set(aml_dir.rglob("*.tif")) | set(aml_dir.rglob("*.tiff")))
        hea_all = sorted(set(hea_dir.rglob("*.tif")) | set(hea_dir.rglob("*.tiff")))
        print(f"📊 Total available: AML={len(aml_all)}, Healthy={len(hea_all)}")

        def take_slice(arr, lo, hi):
            n = len(arr)
            i0 = int(n * lo)
            i1 = int(n * hi)
            return arr[i0:i1]

        if region == "cal":
            aml_pool = take_slice(aml_all, 0.60, 0.70)
            hea_pool = take_slice(hea_all, 0.60, 0.70)
        elif region == "test":
            aml_pool = take_slice(aml_all, 0.70, 1.00)
            hea_pool = take_slice(hea_all, 0.70, 1.00)
        else:
            raise ValueError(f"Unknown region: {region}")

        print(f"📊 Pool sizes[{region}]: AML={len(aml_pool)}, Healthy={len(hea_pool)}")

        # Keep disjoint from another split if requested
        if exclude_paths:
            excl = set(map(str, exclude_paths))
            aml_pool = [p for p in aml_pool if str(p) not in excl]
            hea_pool = [p for p in hea_pool if str(p) not in excl]
            print(f"🔒 After exclusion: AML={len(aml_pool)}, Healthy={len(hea_pool)}")

        # Balanced sample size
        n = min(test_per_class, len(aml_pool), len(hea_pool))
        if n <= 0:
            raise RuntimeError(f"No data available for region={region} after filtering.")

        aml_sel = aml_pool if len(aml_pool) <= n else rng.sample(aml_pool, n)
        hea_sel = hea_pool if len(hea_pool) <= n else rng.sample(hea_pool, n)
        m = min(len(aml_sel), len(hea_sel))
        aml_sel, hea_sel = aml_sel[:m], hea_sel[:m]

        # Build lists
        for p in aml_sel:
            self.paths.append(str(p))
            self.images.append(p)
            self.label_vec.append(1)
            self.labels_dict.append({"extraction_method": 1})
        for p in hea_sel:
            self.paths.append(str(p))
            self.images.append(p)
            self.label_vec.append(0)
            self.labels_dict.append({"extraction_method": 0})

        print(f"✅ Balanced[{region}] — AML={len(aml_sel)}, Healthy={len(hea_sel)}, Total={len(self.images)}")

        # Persist exact file list
        Path("splits").mkdir(exist_ok=True, parents=True)
        list_path = Path(f"splits/balanced_{region}_seed{seed}.txt")
        with list_path.open("w") as f:
            for p in self.images:
                f.write(str(p) + "\n")
        print(f"📝 Saved file list to {list_path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img = tifffile.imread(path)

        # Robust channel handling
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3:
            # If CHW with small C, transpose to HWC
            if img.shape[0] <= 4:
                img = np.transpose(img[:3], (1, 2, 0))
            else:
                # Already HWC? If last dim looks like channels, keep it
                if 1 < img.shape[-1] <= 10:
                    img = img[..., :3] if img.shape[-1] > 3 else img
                else:
                    img = np.transpose(img[:3], (1, 2, 0))

        # Ensure 3 channels
        if img.shape[-1] != 3:
            h, w = img.shape[:2]
            new_img = np.zeros((h, w, 3), dtype=np.float32)
            c = min(3, img.shape[-1])
            new_img[..., :c] = img[..., :c]
            img = new_img

        # Resize & normalize
        img = cv2.resize(img.astype(np.float32), (self.img_size, self.img_size))
        img = img.astype(np.float32)
        mx = float(img.max())
        if mx > 0:
            if mx > 255:      # likely uint16
                img /= 65535.0
            elif mx > 1:      # likely uint8
                img /= 255.0

        # To tensor (CHW)
        img = torch.from_numpy(img).permute(2, 0, 1)

        return {
            "image": img,
            "label": int(self.label_vec[idx]),
            "labels": self.labels_dict[idx],  # legacy compatibility
            "path": self.paths[idx],
        }


# ===================
# Helper / calibration
# ===================
def choose_aml_column_by_auc(cal_sm: np.ndarray, cal_y: np.ndarray) -> int:
    """Pick AML column (0/1) by maximizing calibration AUC (stable & simple)."""
    try:
        auc0 = roc_auc_score(cal_y, cal_sm[:, 0])
        auc1 = roc_auc_score(cal_y, cal_sm[:, 1])
    except Exception:
        # Fallback if AUC fails for some reason
        m1 = cal_sm[cal_y == 1].mean(axis=0) if (cal_y == 1).any() else np.array([0, 0])
        m0 = cal_sm[cal_y == 0].mean(axis=0) if (cal_y == 0).any() else np.array([1, 1])
        d = m1 - m0
        return int(d[1] >= d[0])
    return int(auc1 >= auc0)


def grid_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "accuracy",
    target_recall: float | None = None,
    min_threshold: float = 0.02,
):
    """
    Select a threshold on calibration probabilities.

    method ∈ {"accuracy","youden","f1"} OR use target_recall to select the smallest t with recall ≥ target_recall.
    We also enforce a floor (min_threshold) to avoid degenerate ultra-low thresholds on saturated outputs.
    """
    ts = np.unique(
        np.concatenate(
            [
                np.linspace(1e-5, 0.01, 20),
                np.linspace(0.01, 0.20, 40),
                np.linspace(0.20, 0.99, 80),
                np.percentile(y_prob, np.linspace(0, 100, 41)),
            ]
        )
    )
    ts = ts[ts >= min_threshold]

    def acc_at(t):
        p = (y_prob >= t).astype(int)
        return (p == y_true).mean()

    def youden_at(t):
        p = (y_prob >= t).astype(int)
        tp = ((p == 1) & (y_true == 1)).sum()
        tn = ((p == 0) & (y_true == 0)).sum()
        fp = ((p == 1) & (y_true == 0)).sum()
        fn = ((p == 0) & (y_true == 1)).sum()
        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        return tpr - fpr

    def f1_at(t):
        p = (y_prob >= t).astype(int)
        tp = ((p == 1) & (y_true == 1)).sum()
        fp = ((p == 1) & (y_true == 0)).sum()
        fn = ((p == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        return 2 * prec * rec / (prec + rec + 1e-9)

    if target_recall is not None:
        best = None
        for t in ts:
            p = (y_prob >= t).astype(int)
            tp = ((p == 1) & (y_true == 1)).sum()
            fn = ((p == 0) & (y_true == 1)).sum()
            rec = tp / (tp + fn + 1e-9)
            if rec + 1e-9 >= target_recall:
                score = acc_at(t)  # break ties by accuracy
                if (best is None) or (t < best[0]) or (t == best[0] and score > best[2]):
                    best = (t, rec, score)
        if best is None:
            # fall back to accuracy if recall target impossible
            method = "accuracy"
        else:
            return float(best[0]), "target_recall"

    if method == "youden":
        vals = np.array([youden_at(t) for t in ts])
    elif method == "f1":
        vals = np.array([f1_at(t) for t in ts])
    else:
        method = "accuracy"
        vals = np.array([acc_at(t) for t in ts])

    t = float(ts[np.argmax(vals)])
    return t, method


def fit_temperature_on_cal(logits_list: list[torch.Tensor], labels_list: list[torch.Tensor], device: torch.device):
    """
    Simple 1-parameter temperature scaling on the *extraction* logits.
    We learn T>0 to minimize NLL on calibration set. Return float(T).
    """
    with torch.no_grad():
        logits = torch.cat([z.to(device) for z in logits_list], dim=0).float()
        y = torch.cat([z.to(device) for z in labels_list], dim=0).long()

    # Only 2-way head is used (index 0/1)
    t_raw = torch.nn.Parameter(torch.tensor([0.0], device=device))  # T = softplus(t_raw)+eps
    opt = torch.optim.LBFGS([t_raw], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")
    eps = 1e-6

    ce = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad(set_to_none=True)
        T = torch.nn.functional.softplus(t_raw) + eps
        scaled = logits / T
        loss = ce(scaled, y)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        pass

    with torch.no_grad():
        T = float(torch.nn.functional.softplus(t_raw).item() + eps)
    return max(T, 1e-3)


# ==============
# Main inference
# ==============
def run_inference(config):
    """
    Large, disjoint, balanced inference with:
      • Stable AML column selection (calibration AUC)
      • Optional temperature scaling on calibration logits
      • Robust threshold selection (accuracy/youden/f1 or target_recall)
    """
    from src.models import BD_S8_Model

    torch.manual_seed(42)
    np.random.seed(42)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    device_type = "cuda" if use_cuda else "cpu"

    # Safe batch/precision defaults
    SAFE_BATCH = 64 if use_cuda else 32
    autocast_dtype = torch.float16 if use_cuda else torch.float32

    # Load checkpoint
    model_path = Path("models/last.ckpt")
    if not model_path.exists():
        ckpts = sorted(Path("models").glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError("No checkpoint found in models/")
        model_path = ckpts[-1]
    print(f"📊 Loading: {model_path}")

    model = BD_S8_Model(num_classes=5).to(device).eval()
    ck = torch.load(model_path, map_location=device)
    state = ck.get("state_dict", ck)
    clean = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
    missing = model.load_state_dict(clean, strict=False)
    print("✅ state_dict load:", missing)

    # Config
    inf_cfg = (config or {}).get("inference", {}) if isinstance(config, dict) else {}
    cal_per_class = int(inf_cfg.get("cal_per_class", 4000))
    eval_per_class = int(inf_cfg.get("eval_per_class", 16516))  # use max AML in test slice
    img_size = int(inf_cfg.get("img_size", 512))

    # threshold config
    threshold_method = str(inf_cfg.get("threshold_method", "accuracy"))  # "accuracy", "youden", "f1"
    target_recall = inf_cfg.get("target_recall", None)  # e.g. 0.90 or None
    min_threshold = float(inf_cfg.get("min_threshold", 0.02))  # floor to avoid silly tiny cuts

    # optional temperature scaling
    use_temp_scaling = bool(inf_cfg.get("temperature_scale", False))

    # --------------------------
    # PHASE 1 — Calibration set
    # --------------------------
    print("\n" + "=" * 60)
    print("📊 PHASE 1: THRESHOLD CALIBRATION (stable AML col mapping)")
    print("=" * 60)

    cal_ds = BalancedTestDataset(
        region="cal",
        test_per_class=cal_per_class,
        seed=42,
        img_size=img_size,
    )

    # --------------------------
    # PHASE 2 — Large test set
    # --------------------------
    print("\n" + "=" * 60)
    print("📊 PHASE 2: LARGE-SCALE TEST (disjoint)")
    print("=" * 60)

    eval_ds = BalancedTestDataset(
        region="test",
        test_per_class=eval_per_class,
        seed=43,
        exclude_paths=cal_ds.images,  # ensure disjoint
        img_size=img_size,
    )

    # DataLoaders
    num_workers = 2 if use_cuda else 0
    pin = use_cuda
    cal_loader = DataLoader(cal_ds, batch_size=SAFE_BATCH, shuffle=False, num_workers=num_workers, pin_memory=pin)
    eval_loader = DataLoader(eval_ds, batch_size=SAFE_BATCH, shuffle=False, num_workers=num_workers, pin_memory=pin)

    # ---- Calibration: collect logits for AML column + (optional) temperature
    cal_logits, cal_y = [], []
    with torch.inference_mode(), torch.amp.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_cuda):
        for b in cal_loader:
            x = b["image"].to(device, non_blocking=True)
            y = b["label"]
            logits = model(x)["extraction"]
            cal_logits.append(logits.detach().to(torch.float32).cpu())
            cal_y.append(y.detach().cpu())

    # Temperature scaling (optional)
    if use_temp_scaling:
        T = fit_temperature_on_cal(cal_logits, cal_y, device)
        print(f"🌡️  Temperature learned on CAL: T = {T:.3f}")
        # re-run softmax with scaled logits
        cal_sm = []
        for z in cal_logits:
            sm = torch.softmax((z.to(torch.float32) / T), dim=1).numpy()
            cal_sm.append(sm)
    else:
        T = 1.0
        cal_sm = [torch.softmax(z.to(torch.float32), dim=1).numpy() for z in cal_logits]

    cal_sm = np.concatenate(cal_sm, axis=0)  # (N,2)
    cal_y = np.concatenate([y.numpy() for y in cal_y], axis=0)  # (N,)

    # AML column by AUC (stable)
    aml_idx = choose_aml_column_by_auc(cal_sm, cal_y)
    print(f"🧭 AML softmax column in use (auto): {aml_idx}  (0=first, 1=second)")

    # Debug stats
    try:
        m_pos = cal_sm[cal_y == 1, aml_idx].mean()
        m_neg = cal_sm[cal_y == 0, aml_idx].mean()
        neg_pcts = np.percentile(cal_sm[cal_y == 0, aml_idx], [50, 75, 90, 95])
        pos_pcts = np.percentile(cal_sm[cal_y == 1, aml_idx], [5, 10, 25, 50])
        print(f"   mean AML_prob on CAL: y=1 -> {m_pos:.4f} | y=0 -> {m_neg:.4f}")
        print(f"   NEG pct [50,75,90,95]: {np.round(neg_pcts, 3)}")
        print(f"   POS pct [ 5,10,25,50]: {np.round(pos_pcts, 3)}")
    except Exception:
        pass

    cal_probs = cal_sm[:, aml_idx]

    # Threshold selection on CAL
    thr, used_method = grid_threshold(
        y_true=cal_y,
        y_prob=cal_probs,
        method=threshold_method,
        target_recall=target_recall,
        min_threshold=min_threshold,
    )
    print(f"📊 Threshold selected: {thr:.6f} (method={used_method}, min_threshold={min_threshold})")

    Path("logs").mkdir(exist_ok=True, parents=True)
    Path("logs/threshold.txt").write_text(f"{thr:.6f}\n")
    Path("logs/aml_column.txt").write_text(str(int(aml_idx)))
    if use_temp_scaling:
        Path("logs/temperature.txt").write_text(f"{T:.6f}\n")

    # Cal accuracy @ thr
    cal_pred = (cal_probs >= thr).astype(int)
    cal_acc = (cal_pred == cal_y).mean()
    print(f"✅ Calibration accuracy @ thr {thr:.4f}: {cal_acc:.3f}")

    # ---- Evaluation on large disjoint set
    print(f"\n🔄 Testing on {len(eval_ds)} samples (batch={SAFE_BATCH}, img={img_size}^2)…")
    eval_probs, eval_y = [], []
    with torch.inference_mode(), torch.amp.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_cuda):
        for i, b in enumerate(eval_loader):
            x = b["image"].to(device, non_blocking=True)
            y = b["label"]
            logits = model(x)["extraction"]
            if i == 0:
                sm_prev = torch.softmax((logits.to(torch.float32) / T), dim=1).cpu().numpy()
                print("  softmax[0:5]:", np.round(sm_prev[:5], 3))
            sm = torch.softmax((logits.to(torch.float32) / T), dim=1).cpu().numpy()
            eval_probs.append(sm[:, aml_idx])
            eval_y.append(y.detach().cpu().numpy())
            if use_cuda and (i % 50 == 0):
                torch.cuda.empty_cache()

    eval_probs = np.concatenate(eval_probs, axis=0)
    eval_y = np.concatenate(eval_y, axis=0)

    # Metrics @ chosen thr
    pred = (eval_probs >= thr).astype(int)
    acc = (pred == eval_y).mean()
    tp = int(((pred == 1) & (eval_y == 1)).sum())
    tn = int(((pred == 0) & (eval_y == 0)).sum())
    fp = int(((pred == 1) & (eval_y == 0)).sum())
    fn = int(((pred == 0) & (eval_y == 1)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    bal_acc = balanced_accuracy_score(eval_y, pred)
    mcc = matthews_corrcoef(eval_y, pred)

    # Ranking metrics
    try:
        roc = roc_auc_score(eval_y, eval_probs)
        ap = average_precision_score(eval_y, eval_probs)
        print(f"🏁 ROC-AUC={roc:.3f} | PR-AUC={ap:.3f}")
    except Exception as e:
        print(f"⚠️ AUC computation skipped: {e}")

    print(f"\n🎉 TEST RESULTS (N={len(eval_y)}):")
    print(f"✅ Accuracy: {acc:.3f} ({acc*100:.1f}%)")
    print(f"🟰 Balanced Acc: {bal_acc:.3f}  |  MCC: {mcc:.3f}")
    print(f"📊 Confusion: TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"📈 AML: precision={prec:.3f} recall={rec:.3f} F1={f1:.3f}")

    # Save compact results
    results = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "mcc": float(mcc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "n_samples": int(len(eval_y)),
        "threshold": float(thr),
        "aml_column": int(aml_idx),
        "temperature": float(T),
        "threshold_method": used_method,
        "min_threshold": float(min_threshold),
    }
    with Path("logs/test_results.json").open("w") as f:
        json.dump(results, f, indent=2)
    print("💾 Results saved to logs/test_results.json")

    return pred, eval_y


if __name__ == "__main__":
    cfg = {
        "inference": {
            # dataset sizes
            "cal_per_class": 4000,
            "eval_per_class": 16516,   # full AML coverage in 70–100% slice

            # image + batching
            "img_size": 512,

            # thresholding options:
            #   - choose ONE of the following strategies:
            #     "accuracy", "youden", "f1"  (or set target_recall to a float like 0.90)
            "threshold_method": "accuracy",
            "target_recall": None,
            # avoid absurd tiny thresholds when outputs are saturated
            "min_threshold": 0.02,

            # optional calibration (often stabilizes thresholding)
            "temperature_scale": False,
        }
    }
    run_inference(cfg)
    print("\n✅ Inference completed successfully!")
