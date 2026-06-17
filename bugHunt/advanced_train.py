"""
advanced_train.py
=================
Advanced Bug Bounty Model Training Pipeline.

Features:
  1. COLLECT:   Live data from NVD, GitHub Advisory, CISA KEV, OSV, HackerOne
  2. CURRICULUM: Easy samples first → hard (low confidence) samples amplified
  3. MULTI-TASK: Jointly trains vuln-type + severity + chain-impact heads
  4. AUGMENT:   Random token masking, label smoothing, mixup
  5. STREAM:    Real-time metrics → logs/advanced_metrics.jsonl + dashboard
  6. ITERATE:   Adaptive LR + OneCycleLR scheduler

Usage:
  python advanced_train.py                        # full pipeline
  python advanced_train.py --skip-collect         # use cached data
  python advanced_train.py --watch                # + live dashboard on :5050
  python advanced_train.py --skip-collect --data-percent 5
  python advanced_train.py --epochs 30 --synth-n 20000
"""

import os, sys, json, time, random, argparse, threading, math, hashlib, gzip
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from advanced_model import (AdvancedBugBountyModel, SimpleTokenizer,
                             create_advanced_model, VULN_LABELS, CHAIN_IMPACTS,
                             SEVERITY_LEVELS, NUM_CLASSES, NUM_CHAIN_IMPACTS, NUM_SEVERITY)
from live_data_collector import collect_all
from paths import PROJECT_DIR, RAW_DIR, MODELS_DIR, LOGS_DIR, CHECKPOINTS_DIR

# ─── Paths ────────────────────────────────────────────────────────────────────
CHECKPOINTS  = CHECKPOINTS_DIR
METRICS_FILE = LOGS_DIR / "advanced_metrics.jsonl"

for d in [MODELS_DIR, LOGS_DIR, CHECKPOINTS, RAW_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_TO_IDX = {l: i for i, l in enumerate(VULN_LABELS)}
CHAIN_TO_IDX = {c: i for i, c in enumerate(CHAIN_IMPACTS)}
SEV_TO_IDX   = {s: i for i, s in enumerate(SEVERITY_LEVELS)}

TRAINING_PROFILES = {
    "min": {
        "epochs": 3,
        "batch_size": 8,
        "num_workers": 0,
        "patience": 2,
        "synth_n": 1000,
        "nvd_per_kw": 25,
        "gh_max": 100,
        "osv_per_eco": 25,
        "h1_pages": 1,
        "http_aux_weight": 0.0,
        "label_smoothing": 0.02,
        "log_every_steps": 25,
        "no_periodic_checkpoints": True,
    },
    "default": {},
    "max": {
        "epochs": 60,
        "patience": 20,
        "synth_n": 50000,
        "nvd_per_kw": 1000,
        "gh_max": 10000,
        "osv_per_eco": 1000,
        "h1_pages": 50,
        "sev_weight": 0.5,
        "chain_weight": 1.0,
        "is_chain_weight": 0.5,
        "http_aux_weight": 0.5,
        "label_smoothing": 0.08,
        "min_data_gb": 20.0,
        "bulk_text_percent": 10.0,
        "bulk_http_percent": 100.0,
        "no_periodic_checkpoints": True,
        "log_every_steps": 50,
    },
}

# ─── SSE emitter ─────────────────────────────────────────────────────────────
_sse_queue: list = []
_sse_lock = threading.Lock()

def emit(event: dict):
    event["ts"] = datetime.utcnow().isoformat()
    line = json.dumps(event)
    with open(METRICS_FILE, "a") as f:
        f.write(line + "\n")
    with _sse_lock:
        _sse_queue.append(line)
        if len(_sse_queue) > 5000:
            _sse_queue.pop(0)

def gpu_mem():
    return torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def latest_best_model_path() -> Path:
    """Return the promoted best weights used for reinforcement training."""
    path = MODELS_DIR / "advanced_model_best.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"--resume-latest-best requested, but {path} does not exist. "
            "Train or provide --resume first."
        )
    return path

def raw_data_size_gb() -> float:
    """Measure local raw cache size, using scale metadata for compressed Kaggle packs."""
    total = sum(p.stat().st_size for p in RAW_DIR.glob("*.json*") if p.is_file()) / (1024 ** 3)
    summary_path = RAW_DIR / "scale_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            total = max(total, float(summary.get("total_gb", 0.0) or 0.0))
        except Exception as exc:
            print(f"  Warning: could not parse {summary_path}: {exc}")
    return total

def enforce_min_data_gb(min_gb: float):
    if min_gb <= 0:
        return
    size_gb = raw_data_size_gb()
    print(f"  Raw JSON cache size: {size_gb:.3f} GiB")
    if size_gb < min_gb:
        raise ValueError(
            f"Raw dataset cache is {size_gb:.3f} GiB, below --min-data-gb={min_gb}. "
            "Run scale_dataset.py or add more data before training."
        )


def require_cuda_if_requested(require_cuda: bool):
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--require-cuda was set, but torch.cuda.is_available() is false")


def resolve_raw_file(name: str) -> Path | None:
    candidates = [RAW_DIR / name]
    if not name.endswith(".gz"):
        candidates.append(RAW_DIR / f"{name}.gz")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def require_data_files(required_files: list[str]):
    missing = [name for name in required_files if resolve_raw_file(name) is None]
    if missing:
        raise FileNotFoundError(
            "Required training data files are missing from "
            f"{RAW_DIR}: {', '.join(missing)}"
        )


def cuda_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def training_profile_defaults(profile: str) -> dict:
    defaults = dict(TRAINING_PROFILES[profile])
    if profile == "max":
        vram = cuda_vram_gb()
        defaults["batch_size"] = 64 if vram >= 14 else 16 if vram >= 4 else 8
        defaults["num_workers"] = 4 if vram >= 14 else 2 if vram >= 4 else 0
    return defaults


def explicit_arg_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    explicit = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt in argv or any(arg.startswith(f"{opt}=") for arg in argv):
                explicit.add(action.dest)
    return explicit


def apply_training_profile(args, parser: argparse.ArgumentParser, argv: list[str]):
    explicit = explicit_arg_dests(parser, argv)
    for dest, value in training_profile_defaults(args.training_profile).items():
        if dest not in explicit:
            setattr(args, dest, value)
    return explicit


def normalize_percent(value: float, name: str) -> float:
    if value < 0 or value > 100:
        raise ValueError(f"{name} must be between 0 and 100")
    return float(value)


def normalize_data_selection(args, explicit: set[str]) -> None:
    if args.data_percent is not None:
        args.data_percent = normalize_percent(args.data_percent, "--data-percent")
        if "bulk_text_percent" not in explicit and "load_bulk_text_cache" not in explicit:
            args.bulk_text_percent = args.data_percent
        if "bulk_http_percent" not in explicit and "load_bulk_http_cache" not in explicit:
            args.bulk_http_percent = args.data_percent

    if args.load_bulk_text_cache and "bulk_text_percent" not in explicit:
        args.bulk_text_percent = 100.0
    if args.load_bulk_http_cache and "bulk_http_percent" not in explicit:
        args.bulk_http_percent = 100.0

    args.bulk_text_percent = normalize_percent(args.bulk_text_percent, "--bulk-text-percent")
    args.bulk_http_percent = normalize_percent(args.bulk_http_percent, "--bulk-http-percent")
    args.load_bulk_text_cache = args.bulk_text_percent > 0
    args.load_bulk_http_cache = args.bulk_http_percent > 0


def print_available_training_profiles():
    print("[TRAIN] available training profiles:")
    print("[TRAIN]   min     quick smoke run, no large bulk cache by default")
    print("[TRAIN]   default balanced local/Kaggle run, no large bulk cache by default")
    print("[TRAIN]   max     high-coverage run; reads 10% bulk text and 100% bulk HTTP unless overridden")


def print_run_options(args):
    collection = "cached" if args.skip_collect else "live"
    print("[TRAIN] run options:")
    print(
        f"[TRAIN]   profile={args.training_profile} collection={collection} "
        f"epochs={args.epochs} batch={args.batch_size} workers={args.num_workers}"
    )
    print(
        f"[TRAIN]   weights vuln={args.vuln_weight} severity={args.sev_weight} "
        f"chain={args.chain_weight} is_chain={args.is_chain_weight} "
        f"http_aux={args.http_aux_weight} smoothing={args.label_smoothing}"
    )
    print(
        f"[TRAIN]   data synth_n={args.synth_n} nvd_per_kw={args.nvd_per_kw} "
        f"gh_max={args.gh_max} osv_per_eco={args.osv_per_eco} h1_pages={args.h1_pages} "
        f"bulk_text={args.bulk_text_percent:.4g}% bulk_http={args.bulk_http_percent:.4g}% "
        f"min_data_gb={args.min_data_gb}"
    )


def model_metadata_payload(args, epoch: int | None = None,
                           metrics: dict | None = None) -> dict:
    return {
        "model_name": "AdvancedBugBountyModel",
        "architecture": "http_fusion_multitask",
        "timestamp": datetime.utcnow().isoformat(),
        "epoch": epoch,
        "run_name": args.run_name,
        "training_profile": args.training_profile,
        "n_classes": NUM_CLASSES,
        "n_vuln_classes": NUM_CLASSES,
        "vuln_labels": VULN_LABELS,
        "n_severity": NUM_SEVERITY,
        "severity_labels": SEVERITY_LEVELS,
        "n_chain_impacts": NUM_CHAIN_IMPACTS,
        "chain_impacts": CHAIN_IMPACTS,
        "feature_dim": 64,
        "tokenizer": {"type": "SimpleTokenizer", "vocab_size": 8000, "max_len": 256},
        "metrics": metrics or {},
    }


def write_model_metadata(args, epoch: int | None = None,
                         metrics: dict | None = None) -> dict:
    metadata = model_metadata_payload(args, epoch, metrics)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (MODELS_DIR / f"metadata_{args.run_name}.json").write_text(json.dumps(metadata, indent=2))
    (MODELS_DIR / f"advanced_model_best_{args.run_name}.metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    return metadata


def stable_record_id(text: str) -> str:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=12).hexdigest()
    return f"text_{digest}"

# ─── Dataset ──────────────────────────────────────────────────────────────────

def extract_http_features(req: dict, resp: dict) -> list:
    """Extract a 64-dimensional feature vector from an HTTP request/response."""
    path = (req.get("path", "") or "").lower()
    body = (req.get("body", "") or "").lower()
    method = (req.get("method", "GET") or "GET").upper()
    resp_body = (resp.get("body_snippet", "") or "").lower()
    status = int(resp.get("status_code", 200) or 200)
    text = path + " " + body + " " + resp_body

    features = []
    features += [int(method == m) for m in ["GET", "POST", "PUT", "DELETE", "PATCH"]]
    features += [int(p in text) for p in ["' or", "union select", "sleep(", "drop table", "1=1", "having", "group by"]]
    features += [int(p in text) for p in ["<script", "onerror=", "javascript:", "<svg", "onload=", "alert("]]
    features += [int(p in text) for p in ["169.254.169.254", "localhost", "file://", "dict://", "internal", "gopher://"]]
    features += [int(p in text) for p in ["../", "..\\", "%2e%2e", "etc/passwd", "windows/system"]]
    features += [int(p in text) for p in ["; id", "| whoami", "`id`", "$(whoami)", "cmd="]]
    features += [int(p in text) for p in ["{{7*7}}", "${7*7}", "<%=", "#{7*7}"]]
    features += [int(p in text) for p in ["<!entity", "system \"file", "<!doctype foo"]]
    features += [int(p in resp_body) for p in ["sql syntax", "root:x:0", "ami-id", "uid=", "stack trace", "exception"]]
    features += [int(status == s) for s in [200, 500, 302, 403, 404]]
    features += [
        int("/admin" in path), int("/api" in path), int("id=" in path),
        int("url=" in path), int("/upload" in path), int("debug" in path),
        int(len(path) > 100),
    ]
    features += [
        int("'" in text), int("<" in text), int("{" in text),
        int("--" in text), int("/*" in text),
    ]
    while len(features) < 64:
        features.append(0)
    return features[:64]

class BugBountyDataset(Dataset):
    """
    Multi-task dataset for bug bounty vulnerability detection.
    Each sample: text → (vuln_label, severity, is_chain, chain_impact)
    """

    def __init__(self, records: list, tokenizer: SimpleTokenizer,
                 max_len: int = 256, augment: bool = False):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.augment   = augment
        self.samples   = []

        for rec in records:
            text     = rec.get("text", "") or rec.get("description", "") or ""
            label    = rec.get("label", "benign")
            severity = rec.get("severity", "none")
            is_chain = int(rec.get("is_chain", False))

            if not text.strip():
                continue

            chain_impact = rec.get("chain_impact", None) or "no_chain"
            if chain_impact not in CHAIN_TO_IDX:
                chain_impact = "no_chain"

            if label not in LABEL_TO_IDX:
                continue
            if severity not in SEV_TO_IDX:
                severity = "none"

            cvss = float(rec.get("cvss_score", 0.0) or 0.0)
            http_features = rec.get("http_features")
            has_http_features = bool(
                rec.get("has_http_features") and isinstance(http_features, list)
                and len(http_features) == 64
            )
            if not has_http_features:
                http_features = [0.0] * 64

            self.samples.append({
                "text":         text[:1000],
                "vuln_idx":     LABEL_TO_IDX[label],
                "severity_idx": SEV_TO_IDX[severity],
                "is_chain":     is_chain,
                "chain_idx":    CHAIN_TO_IDX[chain_impact],
                "cvss":         min(cvss / 10.0, 1.0),   # normalize 0-1
                "http_features": [float(v) for v in http_features[:64]],
                "has_http_features": has_http_features,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        text = s["text"]

        # Augmentation: random truncation + token dropout
        if self.augment:
            words = text.split()
            if len(words) > 20:
                start = random.randint(0, len(words) // 4)
                words = words[start:]
            # Random word dropout (10%)
            words = [w for w in words if random.random() > 0.10]
            text = " ".join(words)

        ids  = self.tokenizer.tokenize(text)
        mask = [1 if i != 0 else 0 for i in ids]

        return {
            "token_ids":     torch.tensor(ids,           dtype=torch.long),
            "attention_mask":torch.tensor(mask,          dtype=torch.long),
            "vuln_label":    torch.tensor(s["vuln_idx"], dtype=torch.long),
            "severity_label":torch.tensor(s["severity_idx"], dtype=torch.long),
            "is_chain":      torch.tensor(s["is_chain"], dtype=torch.float),
            "chain_label":   torch.tensor(s["chain_idx"], dtype=torch.long),
            "cvss":          torch.tensor(s["cvss"],     dtype=torch.float),
            "http_features": torch.tensor(s["http_features"], dtype=torch.float),
            "has_http_features": torch.tensor(s["has_http_features"], dtype=torch.bool),
        }


class HTTPDataset(Dataset):
    """Dataset for HTTP feature samples."""

    def __init__(self, records: list):
        self.samples = []
        for rec in records:
            label = rec.get("label", "benign")
            if label not in LABEL_TO_IDX:
                continue
            severity = rec.get("severity", "none")
            if severity not in SEV_TO_IDX:
                severity = "none"
            is_chain = int(rec.get("is_chain", False))
            chain_impact = rec.get("chain_impact") or "no_chain"
            if chain_impact not in CHAIN_TO_IDX:
                chain_impact = "no_chain"

            # Extract 64-dim feature from HTTP sample
            req  = rec.get("request", {}) or {}
            resp = rec.get("response", {}) or {}
            feat = self._extract_features(req, resp)

            self.samples.append({
                "features":      feat,
                "vuln_label":    LABEL_TO_IDX[label],
                "severity_label":SEV_TO_IDX[severity],
                "is_chain":      is_chain,
                "chain_label":   CHAIN_TO_IDX[chain_impact],
            })

    def _extract_features(self, req: dict, resp: dict) -> list:
        """Extract 64-dimensional feature vector from HTTP request/response."""
        return extract_http_features(req, resp)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "http_features":  torch.tensor(s["features"],      dtype=torch.float),
            "vuln_label":     torch.tensor(s["vuln_label"],    dtype=torch.long),
            "severity_label": torch.tensor(s["severity_label"],dtype=torch.long),
            "is_chain":       torch.tensor(s["is_chain"],      dtype=torch.float),
            "chain_label":    torch.tensor(s["chain_label"],   dtype=torch.long),
        }


# ─── Loss ─────────────────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss:
      L = w1*vuln + w2*severity + w3*chain_impact + w4*is_chain_bce
    """

    def __init__(self, vuln_weight=1.0, sev_weight=0.3,
                 chain_weight=0.5, is_chain_weight=0.2,
                 http_aux_weight=0.0,
                 label_smoothing=0.1):
        super().__init__()
        self.vuln_ce   = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.sev_ce    = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.chain_ce  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.http_aux_ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.is_chain  = nn.BCEWithLogitsLoss()
        self.w = (vuln_weight, sev_weight, chain_weight, is_chain_weight)
        self.http_aux_weight = http_aux_weight

    def forward(self, out: dict, batch: dict, device):
        vuln_labels = batch["vuln_label"].to(device)
        vuln_loss  = self.vuln_ce(out["vuln_logits"], vuln_labels)
        sev_loss   = self.sev_ce(out["severity_logits"], batch["severity_label"].to(device))
        chain_loss = self.chain_ce(out["chain_logits"], batch["chain_label"].to(device))
        ic_loss    = self.is_chain(out["is_chain_logit"].squeeze(-1),
                                   batch["is_chain"].to(device))
        total = (self.w[0]*vuln_loss + self.w[1]*sev_loss +
                 self.w[2]*chain_loss + self.w[3]*ic_loss)
        http_aux_loss = vuln_loss.new_tensor(0.0)
        if (
            self.http_aux_weight > 0
            and "http_logits" in out
            and "has_http_features" in batch
        ):
            http_mask = batch["has_http_features"].to(device).bool()
            if http_mask.any():
                http_aux_loss = self.http_aux_ce(
                    out["http_logits"][http_mask], vuln_labels[http_mask]
                )
                total = total + self.http_aux_weight * http_aux_loss
        return total, {
            "vuln": vuln_loss.detach().item(), "sev": sev_loss.detach().item(),
            "chain": chain_loss.detach().item(), "is_chain": ic_loss.detach().item(),
            "http_aux": http_aux_loss.detach().item(),
        }


# ─── Curriculum Learning ──────────────────────────────────────────────────────

class CurriculumSampler:
    """
    Tracks per-sample difficulty (loss) and upweights hard samples.
    Stage 1 (epochs 1-3):   easy samples only (p < median loss)
    Stage 2 (epochs 4-8):   all samples uniform
    Stage 3 (epochs 9+):    hard mining (3x weight for hard samples)
    """

    def __init__(self, n_samples: int):
        self.n = n_samples
        self.losses = np.ones(n_samples, dtype=np.float32)
        self.epoch  = 0

    def get_weights(self) -> np.ndarray:
        if self.epoch < 3:
            # Stage 1: easy first
            threshold = np.median(self.losses)
            w = np.where(self.losses <= threshold, 1.0, 0.1)
        elif self.epoch < 8:
            # Stage 2: uniform
            w = np.ones(self.n)
        else:
            # Stage 3: hard mining
            loss_range = np.ptp(self.losses)
            normalized = (self.losses - self.losses.min()) / (loss_range + 1e-8)
            w = 1.0 + 2.0 * normalized   # hard samples get up to 3x weight
        return (w / w.sum()).astype(np.float32)

    def update(self, indices: list, losses: list):
        for idx, loss in zip(indices, losses):
            self.losses[idx] = 0.9 * self.losses[idx] + 0.1 * loss  # EMA


# ─── Trainer ─────────────────────────────────────────────────────────────────

class AdvancedTrainer:

    def __init__(self, model: AdvancedBugBountyModel, args):
        self.model  = model.to(DEVICE)
        self.args   = args
        self.scaler = GradScaler("cuda") if DEVICE == "cuda" else None
        self.criterion = MultiTaskLoss(
            vuln_weight=args.vuln_weight,
            sev_weight=args.sev_weight,
            chain_weight=args.chain_weight,
            is_chain_weight=args.is_chain_weight,
            http_aux_weight=args.http_aux_weight,
            label_smoothing=args.label_smoothing,
        )

        # AdamW with weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm"]
        params = [
            {"params": [p for n,p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)], "weight_decay": 1e-4},
            {"params": [p for n,p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],  "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(params, lr=args.lr)
        self.best_val_loss = float("inf")
        self.best_val_f1   = 0.0
        self.best_epoch    = 0
        self.patience_ctr  = 0
        self.best_ckpt_path = CHECKPOINTS / f"advanced_best_{args.run_name}.pt"
        self.best_model_path = MODELS_DIR / f"advanced_model_best_{args.run_name}.pt"

    def train_epoch(self, loader, epoch: int, curriculum: CurriculumSampler = None,
                    scheduler=None) -> dict:
        self.model.train()
        total_loss = 0.0
        preds_all, labels_all = [], []
        loss_breakdown = {"vuln": 0, "sev": 0, "chain": 0, "is_chain": 0, "http_aux": 0}

        if curriculum:
            curriculum.epoch = epoch

        pbar = tqdm(loader, desc=f"  [E{epoch+1:02d}] Train", unit="batch",
                    leave=False, dynamic_ncols=True)

        for step, batch in enumerate(pbar):
            tok  = batch["token_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            http_features = batch["http_features"].to(DEVICE)
            http_feature_mask = batch["has_http_features"].to(DEVICE)

            self.optimizer.zero_grad()
            optimizer_stepped = True

            if self.scaler:
                scale_before = self.scaler.get_scale()
                with autocast("cuda"):
                    out = self.model(
                        tok, mask,
                        http_features=http_features,
                        http_feature_mask=http_feature_mask,
                    )
                    loss, breakdown = self.criterion(out, batch, DEVICE)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                optimizer_stepped = self.scaler.get_scale() >= scale_before
            else:
                out = self.model(
                    tok, mask,
                    http_features=http_features,
                    http_feature_mask=http_feature_mask,
                )
                loss, breakdown = self.criterion(out, batch, DEVICE)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            if scheduler is not None and optimizer_stepped:
                scheduler.step()

            loss_value = loss.detach().item()
            total_loss += loss_value
            for k in loss_breakdown:
                loss_breakdown[k] += breakdown[k]

            preds  = out["vuln_logits"].argmax(1).cpu().tolist()
            labels = batch["vuln_label"].tolist()
            preds_all.extend(preds)
            labels_all.extend(labels)

            mem = gpu_mem()
            pbar.set_postfix({"loss": f"{loss_value:.3f}", "mem": f"{mem:.2f}GB"})

            event = {"phase": "train", "type": "batch", "epoch": epoch+1,
                     "step": step, "loss": round(loss_value, 5), "gpu_gb": round(mem, 3),
                     **{f"loss_{k}": round(v/(step+1), 5) for k,v in loss_breakdown.items()}}
            emit(event)
            log_every = max(int(getattr(self.args, "log_every_steps", 100)), 1)
            if step == 0 or (step + 1) % log_every == 0:
                print(
                    f"  [verbose] epoch={epoch+1} step={step+1}/{len(loader)} "
                    f"loss={loss_value:.5f} gpu_gb={mem:.3f}",
                    flush=True,
                )

        n = max(len(loader), 1)
        avg_loss = total_loss / n
        acc = accuracy_score(labels_all, preds_all)
        f1  = f1_score(labels_all, preds_all, average="weighted", zero_division=0)

        return {"loss": avg_loss, "acc": acc, "f1": f1,
                **{f"loss_{k}": v/n for k,v in loss_breakdown.items()}}

    @torch.no_grad()
    def validate(self, loader, epoch: int) -> dict:
        self.model.eval()
        total_loss = 0.0
        preds_all, labels_all = [], []

        for batch in tqdm(loader, desc=f"  [E{epoch+1:02d}] Val  ",
                          unit="batch", leave=False, dynamic_ncols=True):
            tok  = batch["token_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            out  = self.model(
                tok, mask,
                http_features=batch["http_features"].to(DEVICE),
                http_feature_mask=batch["has_http_features"].to(DEVICE),
            )
            loss, _ = self.criterion(out, batch, DEVICE)

            total_loss += loss.detach().item()
            preds_all.extend(out["vuln_logits"].argmax(1).cpu().tolist())
            labels_all.extend(batch["vuln_label"].tolist())

        avg_loss = total_loss / max(len(loader), 1)
        acc = accuracy_score(labels_all, preds_all)
        f1  = f1_score(labels_all, preds_all, average="weighted", zero_division=0)
        precision = f1_score(labels_all, preds_all, average="macro", zero_division=0)

        emit({"phase": "val", "type": "epoch", "epoch": epoch+1,
              "loss": round(avg_loss, 5), "acc": round(acc, 5),
              "f1": round(f1, 5), "macro_f1": round(precision, 5)})

        return {"loss": avg_loss, "acc": acc, "f1": f1, "macro_f1": precision}

    @torch.no_grad()
    def test(self, loader) -> dict:
        self.model.eval()
        preds_all, labels_all = [], []

        for batch in tqdm(loader, desc="  [TEST]", unit="batch",
                          leave=False, dynamic_ncols=True):
            tok  = batch["token_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            out  = self.model(
                tok, mask,
                http_features=batch["http_features"].to(DEVICE),
                http_feature_mask=batch["has_http_features"].to(DEVICE),
            )
            preds_all.extend(out["vuln_logits"].argmax(1).cpu().tolist())
            labels_all.extend(batch["vuln_label"].tolist())

        acc = accuracy_score(labels_all, preds_all)
        f1w = f1_score(labels_all, preds_all, average="weighted", zero_division=0)
        f1m = f1_score(labels_all, preds_all, average="macro", zero_division=0)

        # Per-class report (only classes present in test set)
        present = sorted(set(labels_all))
        names   = [VULN_LABELS[i] for i in present]
        report  = classification_report(labels_all, preds_all, labels=present,
                                        target_names=names, output_dict=True,
                                        zero_division=0)
        return {"acc": acc, "f1_weighted": f1w, "f1_macro": f1m,
                "per_class": report}

    def fit(self, train_loader, val_loader, test_loader=None) -> dict:
        args = self.args
        print(f"\n{'─'*70}")
        print(f"  ADVANCED BUG BOUNTY MODEL TRAINING")
        print(f"  Epochs={args.epochs}  LR={args.lr}  Batch={args.batch_size}")
        print(f"  Classes={NUM_CLASSES}  Device={DEVICE.upper()}")
        print(f"{'─'*70}")

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=args.lr,
            total_steps=args.epochs * len(train_loader),
            pct_start=0.1,
            anneal_strategy="cos",
        )

        curriculum = CurriculumSampler(len(train_loader.dataset))
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
                   "train_f1": [], "val_f1": []}

        for epoch in range(args.epochs):
            t0 = time.time()
            tr = self.train_epoch(train_loader, epoch, curriculum, scheduler)
            vl = self.validate(val_loader, epoch)

            elapsed = time.time() - t0
            lr_now  = self.optimizer.param_groups[0]["lr"]

            history["train_loss"].append(tr["loss"])
            history["val_loss"].append(vl["loss"])
            history["train_acc"].append(tr["acc"])
            history["val_acc"].append(vl["acc"])
            history["train_f1"].append(tr["f1"])
            history["val_f1"].append(vl["f1"])

            stage = "easy" if epoch < 3 else "uniform" if epoch < 8 else "hard-mining"
            print(
                f"  Ep {epoch+1:02d}/{args.epochs} │"
                f" tr_loss={tr['loss']:.4f} tr_acc={tr['acc']:.4f} tr_f1={tr['f1']:.4f} │"
                f" vl_loss={vl['loss']:.4f} vl_acc={vl['acc']:.4f} vl_f1={vl['f1']:.4f} │"
                f" lr={lr_now:.2e} [{stage}] {elapsed:.1f}s"
            )

            improved = vl["loss"] < self.best_val_loss
            if improved:
                self.best_val_f1   = vl["f1"]
                self.best_val_loss = vl["loss"]
                self.best_epoch    = epoch + 1
                self.patience_ctr  = 0
                metadata = write_model_metadata(args, epoch + 1, {
                    "val_loss": vl["loss"],
                    "val_f1": vl["f1"],
                    "val_acc": vl["acc"],
                })
                ckpt = {
                    "epoch": epoch + 1,
                    "val_loss": vl["loss"],
                    "val_f1": vl["f1"],
                    "val_acc": vl["acc"],
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "n_classes": NUM_CLASSES,
                    "n_vuln_classes": NUM_CLASSES,
                    "vuln_labels": VULN_LABELS,
                    "n_severity": NUM_SEVERITY,
                    "severity_labels": SEVERITY_LEVELS,
                    "n_chain_impacts": NUM_CHAIN_IMPACTS,
                    "chain_impacts": CHAIN_IMPACTS,
                    "model_name": "AdvancedBugBountyModel",
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": metadata,
                    "label_map": LABEL_TO_IDX,
                    "run_name": args.run_name,
                }
                torch.save(ckpt, self.best_ckpt_path)
                torch.save(ckpt, self.best_model_path)
                if args.update_default_best:
                    torch.save(ckpt, CHECKPOINTS / "advanced_best.pt")
                    torch.save(ckpt, MODELS_DIR / "advanced_model_best.pt")
                print(
                    f"  ✓ Best checkpoint  (val_loss={vl['loss']:.4f}, "
                    f"val_f1={vl['f1']:.4f}, "
                    f"epoch={epoch+1}, run={args.run_name})"
                )
            else:
                self.patience_ctr += 1
                if self.patience_ctr >= args.patience:
                    print(f"\n  ⏹  Early stopping at epoch {epoch+1}")
                    break

            # Periodic checkpoint
            if (not args.no_periodic_checkpoints) and (epoch + 1) % 5 == 0:
                torch.save({
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "n_classes": NUM_CLASSES,
                    "n_vuln_classes": NUM_CLASSES,
                    "vuln_labels": VULN_LABELS,
                    "n_severity": NUM_SEVERITY,
                    "severity_labels": SEVERITY_LEVELS,
                    "n_chain_impacts": NUM_CHAIN_IMPACTS,
                    "chain_impacts": CHAIN_IMPACTS,
                    "model_name": "AdvancedBugBountyModel",
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": model_metadata_payload(args, epoch + 1, {
                        "train_loss": tr["loss"],
                        "val_loss": vl["loss"],
                        "val_f1": vl["f1"],
                    }),
                    "run_name": args.run_name,
                }, MODELS_DIR / f"advanced_ep{epoch+1}.pt")

        if self.best_ckpt_path.exists():
            best = torch.load(self.best_ckpt_path, weights_only=True, map_location=DEVICE)
            self.model.load_state_dict(best["model_state"], strict=False)
            print(
                f"\n  Reloaded best checkpoint from epoch "
                f"{best.get('epoch', self.best_epoch)} before test/final save"
            )

        # Final test
        test_results = {}
        if test_loader:
            print("\n  Running test set evaluation…")
            test_results = self.test(test_loader)
            print(f"  Test acc={test_results['acc']:.4f}"
                  f" weighted_f1={test_results['f1_weighted']:.4f}"
                  f" macro_f1={test_results['f1_macro']:.4f}")

        eval_report = {
            "run_name": args.run_name,
            "best_val_f1": self.best_val_f1,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "history": history,
            "test": test_results,
            "args": vars(args),
        }
        eval_path = LOGS_DIR / f"advanced_eval_{args.run_name}.json"
        eval_path.write_text(json.dumps(eval_report, indent=2))
        print(f"  Eval report -> {eval_path}")

        final_metadata = write_model_metadata(args, self.best_epoch, {
            "best_val_loss": self.best_val_loss,
            "best_val_f1": self.best_val_f1,
            "test_acc": test_results.get("acc", 0),
            "test_f1_weighted": test_results.get("f1_weighted", 0),
            "test_f1_macro": test_results.get("f1_macro", 0),
        })
        torch.save({
            "epoch": self.best_epoch,
            "val_loss": self.best_val_loss,
            "val_f1": self.best_val_f1,
            "model_state": self.model.state_dict(),
            "n_classes": NUM_CLASSES,
            "n_vuln_classes": NUM_CLASSES,
            "vuln_labels": VULN_LABELS,
            "n_severity": NUM_SEVERITY,
            "severity_labels": SEVERITY_LEVELS,
            "n_chain_impacts": NUM_CHAIN_IMPACTS,
            "chain_impacts": CHAIN_IMPACTS,
            "model_name": "AdvancedBugBountyModel",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": final_metadata,
            "run_name": args.run_name,
        }, MODELS_DIR / "advanced_model_final.pt")
        emit({"phase": "done", "type": "summary",
              "best_val_f1": self.best_val_f1,
              "best_epoch": self.best_epoch,
              "test_acc": test_results.get("acc", 0),
              "test_f1_weighted": test_results.get("f1_weighted", 0),
              "test_f1_macro": test_results.get("f1_macro", 0)})

        return {"history": history, "test": test_results}


# ─── Data loading ─────────────────────────────────────────────────────────────

def iter_json_array(path: Path, max_bytes: int | None = None):
    """
    Stream records from a top-level JSON array without loading the full file.
    This keeps multi-GB generated datasets usable on workstation RAM.
    """
    in_array = False
    in_string = False
    escaped = False
    depth = 0
    buf = []
    bytes_read = 0
    budget_reached = False

    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            if max_bytes is not None:
                bytes_read += len(chunk)
                if bytes_read >= max_bytes:
                    budget_reached = True

            for ch in chunk:
                if not in_array:
                    if ch == "[":
                        in_array = True
                    continue

                if depth == 0:
                    if ch == "{":
                        depth = 1
                        buf = [ch]
                        in_string = False
                        escaped = False
                    elif ch == "]":
                        return
                    else:
                        continue
                    continue

                buf.append(ch)

                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        yield json.loads("".join(buf))
                        buf = []
                        if budget_reached:
                            return


def percent_byte_budget(path: Path, percent: float) -> int | None:
    if percent >= 100:
        return None
    return max(1, math.ceil(path.stat().st_size * (percent / 100.0)))


def compact_text_record(rec: dict) -> dict:
    text = (
        rec.get("text")
        or rec.get("description")
        or f"{rec.get('title', '')} {rec.get('summary', '')}"
        or ""
    )
    return {
        "id": rec.get("id") or rec.get("cve_id") or stable_record_id(text),
        "text": text[:4000],
        "label": rec.get("label", "benign"),
        "severity": rec.get("severity", "none"),
        "is_chain": rec.get("is_chain", False),
        "chain_impact": rec.get("chain_impact") or "no_chain",
        "cvss_score": rec.get("cvss_score", 0.0),
        "source": rec.get("source", "unknown"),
    }


def load_all_records(bulk_percent: float = 0.0) -> list:
    """Load all collected records from data/raw/."""
    all_records = []
    base_files = [
        "nvd_cves.json", "github_advisories.json", "cisa_kev.json",
        "osv_vulns.json", "hackerone_reports.json", "patt_payloads.json",
        "unified_dataset.json",
    ]
    files: list[tuple[Path, float | None]] = []
    for name in base_files:
        p = resolve_raw_file(name)
        if p:
            files.append((p, None))
    # Bulk scaled datasets (20GB+ pipeline). Prefer compressed variants when present.
    bulk_candidates = sorted(RAW_DIR.glob("bulk_text_records.json*"))
    bulk_candidates.extend(sorted(RAW_DIR.glob("bulk_text_extra_*.json*")))
    bulk_files = [p for p in bulk_candidates if p.is_file()]
    if bulk_percent > 0:
        files.extend((p, bulk_percent) for p in bulk_files)
    elif bulk_files:
        total_gb = sum(p.stat().st_size for p in bulk_files) / (1024 ** 3)
        print(f"  Bulk text cache present ({total_gb:.2f} GiB); selected 0%, skipping")
    seen = set()
    for p, percent in files:
        size_mb = p.stat().st_size / (1024 * 1024)
        max_bytes = percent_byte_budget(p, percent) if percent is not None else None
        if percent is None or percent >= 100:
            suffix = ""
        else:
            suffix = f" [{percent:.4g}% ~= {max_bytes / (1024 * 1024):.1f} MB scan budget]"
        print(f"  Loading {p.name} ({size_mb:.1f} MB){suffix}...", flush=True)
        added = 0
        total = 0
        for r in iter_json_array(p, max_bytes=max_bytes):
            total += 1
            r = compact_text_record(r)
            rid = r.get("id", "") or stable_record_id(r.get("text", ""))
            if rid not in seen:
                seen.add(rid)
                all_records.append(r)
                added += 1
            if total % 5000 == 0:
                print(f"    {p.name}: {total:,} scanned, {added:,} unique", flush=True)
        print(f"  Loaded {total:>6} records from {p.name} (+{added} unique)")

    print(f"  Total unique records: {len(all_records):,}")
    return all_records


def load_http_records(bulk_percent: float = 0.0) -> list:
    records = []
    paths: list[tuple[Path, float | None]] = []
    advanced_http = resolve_raw_file("advanced_http_samples.json")
    if advanced_http:
        paths.append((advanced_http, None))
    bulk_http = resolve_raw_file("bulk_http_samples.json")
    if bulk_percent > 0 and bulk_http:
        paths.append((bulk_http, bulk_percent))
    elif bulk_http:
        size_gb = bulk_http.stat().st_size / (1024 ** 3)
        print(f"  Bulk HTTP cache present ({size_gb:.2f} GB); selected 0%, skipping")

    for p, percent in paths:
        if p.exists():
            max_bytes = percent_byte_budget(p, percent) if percent is not None else None
            if percent is None or percent >= 100:
                suffix = ""
            else:
                suffix = f" [{percent:.4g}% ~= {max_bytes / (1024 * 1024):.1f} MB scan budget]"
            print(f"  Loading HTTP from {p.name}{suffix}…", flush=True)
            records.extend(iter_json_array(p, max_bytes=max_bytes))
    if records:
        print(f"  Loaded {len(records):,} HTTP samples total")
    return records


def http_records_to_text(records: list) -> list:
    converted = []
    for idx, rec in enumerate(records):
        label = rec.get("label", "benign")
        if label not in LABEL_TO_IDX:
            continue
        req = rec.get("request", {}) or {}
        resp = rec.get("response", {}) or {}
        http_features = extract_http_features(req, resp)
        text = (
            f"HTTP {req.get('method', 'GET')} {req.get('path', '')} "
            f"headers={json.dumps(req.get('headers', {}))[:300]} "
            f"body={str(req.get('body', ''))[:1000]} "
            f"response_status={resp.get('status_code', 200)} "
            f"response={str(resp.get('body_snippet', ''))[:1000]}"
        )
        converted.append({
            "id": rec.get("id", f"http_{idx}"),
            "source": rec.get("source", "http_sample"),
            "text": text[:4000],
            "label": label,
            "severity": rec.get("severity", "none"),
            "is_chain": rec.get("is_chain", False),
            "chain_impact": rec.get("chain_impact") or "no_chain",
            "cvss_score": rec.get("cvss_score", 0.0),
            "http_features": http_features,
            "has_http_features": True,
        })
    return converted


def build_dataloaders(text_records: list, http_records: list,
                      tokenizer: SimpleTokenizer, args) -> tuple:
    """Split and build all dataloaders."""
    if http_records:
        http_text = http_records_to_text(http_records)
        text_records = text_records + http_text
        print(f"  Added {len(http_text):,} HTTP-derived text samples")

    rng = random.Random(args.seed)
    rng.shuffle(text_records)
    n = len(text_records)
    tr_n = int(0.80 * n)
    vl_n = int(0.10 * n)

    tr_recs = text_records[:tr_n]
    vl_recs = text_records[tr_n:tr_n+vl_n]
    te_recs = text_records[tr_n+vl_n:]

    tr_ds = BugBountyDataset(tr_recs, tokenizer, augment=True)
    vl_ds = BugBountyDataset(vl_recs, tokenizer, augment=False)
    te_ds = BugBountyDataset(te_recs, tokenizer, augment=False)

    if not tr_ds or not vl_ds or not te_ds:
        raise ValueError(
            f"Not enough usable records after filtering: "
            f"train={len(tr_ds)} val={len(vl_ds)} test={len(te_ds)}"
        )

    # Class balancing weights for the filtered training dataset.
    idx_to_label = {v: k for k, v in LABEL_TO_IDX.items()}
    label_counts = Counter(idx_to_label[s["vuln_idx"]] for s in tr_ds.samples)
    max_count = max(label_counts.values()) if label_counts else 1
    sample_weights = [
        max_count / max(label_counts[idx_to_label[s["vuln_idx"]]], 1)
        for s in tr_ds.samples
    ]

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(tr_ds), replacement=True,
        generator=generator,
    )

    workers = max(int(args.num_workers), 0)
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": (DEVICE == "cuda"),
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler,
                           generator=generator, **loader_kwargs)
    vl_loader = DataLoader(vl_ds, batch_size=args.batch_size, shuffle=False,
                           generator=generator, **loader_kwargs)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                           generator=generator, **loader_kwargs)

    print(f"  Dataloaders: train={len(tr_ds)} val={len(vl_ds)} test={len(te_ds)}")
    print(f"  Classes present: {len(label_counts)}/{NUM_CLASSES}")
    print(f"  Label distribution: {dict(sorted(label_counts.items(), key=lambda x:-x[1])[:8])}")

    return tr_loader, vl_loader, te_loader


def write_dataset_manifest(text_records: list, http_records: list, args):
    label_counts = Counter(r.get("label", "unknown") for r in text_records)
    http_label_counts = Counter(r.get("label", "unknown") for r in http_records)
    manifest = {
        "raw_dir": str(RAW_DIR),
        "raw_size_gb": raw_data_size_gb(),
        "text_records": len(text_records),
        "http_records": len(http_records),
        "text_label_counts_top": dict(label_counts.most_common(20)),
        "http_label_counts_top": dict(http_label_counts.most_common(20)),
        "required_files": args.require_data_file,
        "bulk_text_percent": args.bulk_text_percent,
        "bulk_http_percent": args.bulk_http_percent,
        "synthetic_fallback_disabled": args.no_synthetic_fallback,
    }
    path = LOGS_DIR / f"dataset_manifest_{args.run_name}.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(f"  Dataset manifest -> {path}")


# ─── Dashboard ────────────────────────────────────────────────────────────────

def start_dashboard(port: int = 5050):
    try:
        from flask import Flask, Response, stream_with_context
    except ImportError:
        print("[WATCH] pip install flask")
        return

    app = Flask(__name__)

    DASH_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Advanced Bug Bounty Model — Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root{--bg:#0d1117;--surface:#161b22;--border:#30363d;--accent:#58a6ff;--green:#3fb950;--orange:#f0883e;--purple:#bc8cff;--red:#f85149;--text:#c9d1d9;--dim:#8b949e}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}
  header{background:linear-gradient(90deg,#1a1f2e,#0d1117);border-bottom:1px solid var(--border);padding:16px 28px;display:flex;align-items:center;gap:14px}
  header h1{font-size:1.1rem;font-weight:700;color:var(--accent);letter-spacing:.02em}
  .badge{padding:3px 10px;border-radius:20px;font-size:.7rem;font-weight:700;background:#0d4429;color:var(--green);border:1px solid #2ea043;animation:pulse 1.6s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
  .topbar{display:flex;gap:20px;padding:10px 28px;background:var(--surface);border-bottom:1px solid var(--border);font-size:.78rem;color:var(--dim);flex-wrap:wrap}
  .topbar b{color:var(--text)}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:20px;padding:24px 28px}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px;box-shadow:0 4px 20px rgba(0,0,0,.4)}
  .card h2{font-size:.78rem;font-weight:700;color:var(--dim);margin-bottom:14px;text-transform:uppercase;letter-spacing:.07em}
  .metrics-row{display:flex;gap:12px;margin-bottom:18px;flex-wrap:wrap}
  .metric{flex:1 1 100px;background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:10px 14px}
  .metric .lbl{font-size:.68rem;color:var(--dim);text-transform:uppercase;letter-spacing:.05em}
  .metric .val{font-size:1.55rem;font-weight:700;margin-top:3px}
  .blue{color:var(--accent)}.green{color:var(--green)}.orange{color:var(--orange)}.purple{color:var(--purple)}
  .chart-wrap{position:relative;height:200px}
  .log-box{background:#010409;border:1px solid var(--border);border-radius:6px;height:180px;overflow-y:auto;font-family:'Courier New',monospace;font-size:.72rem;padding:8px 12px;color:var(--dim)}
  .log-box .e{padding:1px 0}.log-box .train{color:var(--accent)}.log-box .val{color:var(--green)}.log-box .done{color:var(--purple)}
  .gpu-wrap{height:20px;background:var(--bg);border-radius:10px;border:1px solid var(--border);overflow:hidden;margin-top:6px}
  .gpu-bar{height:100%;background:linear-gradient(90deg,#1f6feb,#58a6ff);border-radius:10px;transition:width .4s}
  .progress-section{padding:0 28px 24px}
  .stage-bar{display:flex;gap:8px;margin-top:10px}
  .stage{flex:1;padding:8px;border-radius:6px;border:1px solid var(--border);text-align:center;font-size:.72rem;font-weight:600;color:var(--dim);background:var(--bg)}
  .stage.active{border-color:var(--accent);color:var(--accent);background:#1f6feb15}
  .stage.done{border-color:var(--green);color:var(--green);background:#3fb95015}
</style>
</head>
<body>
<header>
  <h1>🛡 Advanced Bug Bounty Model — Training Dashboard</h1>
  <span class="badge" id="liveTag">● LIVE</span>
</header>
<div class="topbar">
  <span>Events: <b id="evtCount">0</b></span>
  <span>Epoch: <b id="curEpoch">—</b></span>
  <span>Phase: <b id="curPhase">—</b></span>
  <span>Classes: <b>34</b></span>
  <span>Last update: <b id="lastTs">—</b></span>
</div>

<div class="progress-section">
  <div style="font-size:.78rem;color:var(--dim);margin-top:14px;margin-bottom:6px;">Curriculum Stage</div>
  <div class="stage-bar">
    <div class="stage" id="s1">📘 Easy (ep 1-3)</div>
    <div class="stage" id="s2">⚖️ Uniform (ep 4-8)</div>
    <div class="stage" id="s3">🔥 Hard Mining (ep 9+)</div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>Live Metrics</h2>
    <div class="metrics-row">
      <div class="metric"><div class="lbl">Train Loss</div><div class="val blue" id="mTrLoss">—</div></div>
      <div class="metric"><div class="lbl">Val Loss</div><div class="val orange" id="mVlLoss">—</div></div>
      <div class="metric"><div class="lbl">Val Acc</div><div class="val green" id="mVlAcc">—</div></div>
      <div class="metric"><div class="lbl">Val F1</div><div class="val purple" id="mVlF1">—</div></div>
    </div>
    <div style="font-size:.73rem;color:var(--dim);margin-bottom:4px">GPU Memory</div>
    <div class="gpu-wrap"><div class="gpu-bar" id="gpuBar" style="width:0%"></div></div>
    <div style="font-size:.7rem;color:var(--dim);margin-top:4px" id="gpuTxt">0.00 / 4.00 GB</div>
  </div>
  <div class="card">
    <h2>Loss Curves</h2>
    <div class="chart-wrap"><canvas id="lossChart"></canvas></div>
  </div>
  <div class="card">
    <h2>Accuracy & F1</h2>
    <div class="chart-wrap"><canvas id="accChart"></canvas></div>
  </div>
  <div class="card">
    <h2>Live Event Log</h2>
    <div class="log-box" id="logBox"></div>
  </div>
</div>

<script>
let evtCount=0;
const hist={tloss:[],vloss:[],tacc:[],vacc:[],tf1:[],vf1:[]};
const GPU_TOTAL=4.0;

function chart(id,datasets,yLabel=''){
  return new Chart(document.getElementById(id),{
    type:'line',data:{labels:[],datasets},
    options:{responsive:true,maintainAspectRatio:false,animation:false,
      plugins:{legend:{labels:{color:'#8b949e',boxWidth:12}}},
      scales:{x:{ticks:{color:'#8b949e',maxTicksLimit:10},grid:{color:'#21262d'}},
              y:{ticks:{color:'#8b949e'},grid:{color:'#21262d'},title:{display:!!yLabel,text:yLabel,color:'#8b949e'}}}}
  });
}

const lossC=chart('lossChart',[
  {label:'Train Loss',data:[],borderColor:'#58a6ff',backgroundColor:'#58a6ff22',borderWidth:2,tension:.3,pointRadius:2},
  {label:'Val Loss',  data:[],borderColor:'#f0883e',backgroundColor:'#f0883e22',borderWidth:2,tension:.3,pointRadius:2},
],'Loss');

const accC=chart('accChart',[
  {label:'Val Acc', data:[],borderColor:'#3fb950',backgroundColor:'#3fb95022',borderWidth:2,tension:.3,pointRadius:2},
  {label:'Val F1',  data:[],borderColor:'#bc8cff',backgroundColor:'#bc8cff22',borderWidth:2,tension:.3,pointRadius:2},
],'Score');

function upd(){
  const n=Math.max(hist.tloss.length,hist.vloss.length);
  const labels=Array.from({length:n},(_,i)=>`E${i+1}`);
  lossC.data.labels=labels; lossC.data.datasets[0].data=hist.tloss; lossC.data.datasets[1].data=hist.vloss;
  accC.data.labels=labels;  accC.data.datasets[0].data=hist.vacc;   accC.data.datasets[1].data=hist.vf1;
  lossC.update('none'); accC.update('none');
}

const log=document.getElementById('logBox');
function lg(msg,cls=''){const d=document.createElement('div');d.className='e '+cls;d.textContent=new Date().toTimeString().slice(0,8)+' '+msg;log.appendChild(d);if(log.children.length>400)log.removeChild(log.firstChild);log.scrollTop=log.scrollHeight;}

function updateStage(ep){
  ['s1','s2','s3'].forEach(id=>{document.getElementById(id).className='stage'});
  if(ep<=3)document.getElementById('s1').className='stage active';
  else if(ep<=8)document.getElementById('s2').className='stage active';
  else document.getElementById('s3').className='stage active';
  if(ep>3)document.getElementById('s1').className='stage done';
  if(ep>8)document.getElementById('s2').className='stage done';
}

const es=new EventSource('/stream');
es.onmessage=function(e){
  const ev=JSON.parse(e.data);
  evtCount++;
  document.getElementById('evtCount').textContent=evtCount;
  document.getElementById('lastTs').textContent=new Date().toTimeString().slice(0,8);
  document.getElementById('curPhase').textContent=ev.phase||'—';
  if(ev.epoch){document.getElementById('curEpoch').textContent=ev.epoch;updateStage(ev.epoch);}

  if(ev.type==='batch'&&ev.phase==='train'){
    document.getElementById('mTrLoss').textContent=ev.loss?.toFixed(4)??'—';
    if(ev.gpu_gb!=null){const p=Math.min(ev.gpu_gb/GPU_TOTAL*100,100);document.getElementById('gpuBar').style.width=p+'%';document.getElementById('gpuTxt').textContent=`${ev.gpu_gb.toFixed(2)} / ${GPU_TOTAL.toFixed(1)} GB`;}
    lg(`train ep${ev.epoch} step${ev.step} loss=${ev.loss?.toFixed(4)}`,  'train');
  }
  if(ev.type==='epoch'&&ev.phase==='val'){
    if(ev.loss!=null){hist.vloss.push(ev.loss);if(hist.tloss.length<hist.vloss.length)hist.tloss.push(ev.loss);}
    if(ev.acc!=null){hist.vacc.push(ev.acc);document.getElementById('mVlAcc').textContent=(ev.acc*100).toFixed(1)+'%';}
    if(ev.f1!=null){hist.vf1.push(ev.f1);document.getElementById('mVlF1').textContent=ev.f1?.toFixed(4);}
    if(ev.loss!=null)document.getElementById('mVlLoss').textContent=ev.loss?.toFixed(4);
    upd();
    lg(`val ep${ev.epoch} loss=${ev.loss?.toFixed(4)} acc=${ev.acc?.toFixed(4)} f1=${ev.f1?.toFixed(4)} macro=${ev.macro_f1?.toFixed(4)}`,'val');
  }
  if(ev.type==='summary')lg(`DONE best_f1=${ev.best_val_f1?.toFixed(4)} test_acc=${ev.test_acc?.toFixed(4)}`,'done');
};
es.onerror=function(){document.getElementById('liveTag').textContent='● DISCONNECTED';document.getElementById('liveTag').style.cssText='background:#3d1c1c;color:#f85149;border-color:#da3633';};
</script>
</body>
</html>"""

    @app.route("/")
    def index():
        return DASH_HTML

    @app.route("/stream")
    def stream():
        def gen():
            last = 0
            while True:
                with _sse_lock:
                    new  = _sse_queue[last:]
                    last = len(_sse_queue)
                for ev in new:
                    yield f"data: {ev}\n\n"
                time.sleep(0.15)
        return Response(stream_with_context(gen()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    t = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=port,
                                                 debug=False, use_reloader=False), daemon=True)
    t.start()
    print(f"\n  🌐 Dashboard: http://localhost:{port}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(
        description="Advanced Bug Bounty Model — Live Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--training-profile", choices=sorted(TRAINING_PROFILES), default="default",
                   help="Preset for min/default/max training intensity; explicit flags override preset values")
    p.add_argument("--print-options", action="store_true",
                   help="Print profile descriptions and selected run options before running")
    p.add_argument("--dry-run", action="store_true",
                   help="Print resolved options and exit before collection/training")
    p.add_argument("--epochs",       type=int,   default=25)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--num-workers",  type=int,   default=0,
                   help="DataLoader workers; raise after checking RAM headroom")
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--patience",     type=int,   default=6)
    p.add_argument("--seed",         type=int,   default=1337)
    p.add_argument("--run-name",     type=str,   default=None,
                   help="Stable suffix for run-specific checkpoints")
    p.add_argument("--update-default-best", action="store_true",
                   help="Also write models/advanced_model_best.pt on improvement")
    p.add_argument("--no-periodic-checkpoints", action="store_true",
                   help="Do not write advanced_epN.pt intermediate model files")
    p.add_argument("--vuln-weight",     type=float, default=1.0)
    p.add_argument("--sev-weight",      type=float, default=0.3)
    p.add_argument("--chain-weight",    type=float, default=0.5)
    p.add_argument("--is-chain-weight", type=float, default=0.2)
    p.add_argument("--http-aux-weight", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--synth-n",      type=int,   default=15000, help="Synthetic HTTP samples")
    p.add_argument("--nvd-per-kw",   type=int,   default=300,   help="NVD CVEs per keyword")
    p.add_argument("--gh-max",       type=int,   default=2000,  help="GitHub advisories")
    p.add_argument("--osv-per-eco",  type=int,   default=200,   help="OSV per ecosystem")
    p.add_argument("--h1-pages",     type=int,   default=10,    help="HackerOne pages")
    p.add_argument("--skip-collect", action="store_true", help="Use cached data")
    p.add_argument("--watch",        action="store_true", help="Live dashboard")
    p.add_argument("--port",         type=int,   default=5050)
    p.add_argument("--github-token", type=str,   default=None,  help="GitHub PAT (optional, 5x rate limit)")
    p.add_argument("--nvd-api-key",  type=str,   default=None,  help="NVD API key (optional)")
    p.add_argument("--resume",       type=str,   default=None,  help="Resume from checkpoint path")
    p.add_argument("--resume-latest-best", action="store_true",
                   help="Reinforce training from models/advanced_model_best.pt unless --resume is set")
    p.add_argument("--min-data-gb",   type=float, default=0.0,
                   help="Require at least this many GiB of raw JSON data before training")
    p.add_argument("--data-percent",  type=float, default=None,
                   help="Percent of large bulk caches to include; 0 skips, 100 streams all")
    p.add_argument("--bulk-text-percent", type=float, default=0.0,
                   help="Percent of bulk_text_records.json* to stream")
    p.add_argument("--bulk-http-percent", type=float, default=0.0,
                   help="Percent of bulk_http_samples.json to stream")
    p.add_argument("--load-bulk-text-cache", action="store_true",
                   help="Deprecated alias for --bulk-text-percent 100")
    p.add_argument("--load-bulk-http-cache", action="store_true",
                   help="Deprecated alias for --bulk-http-percent 100")
    p.add_argument("--no-synthetic-fallback", action="store_true",
                   help="Fail if cached records are missing instead of generating synthetic data")
    p.add_argument("--require-cuda", action="store_true",
                   help="Fail fast unless CUDA is available")
    p.add_argument("--require-data-file", action="append", default=[],
                   help="Raw data file that must exist before training; repeatable")
    p.add_argument("--log-every-steps", type=int, default=100,
                   help="Print verbose batch progress every N optimizer steps")
    args = p.parse_args(argv)
    explicit = apply_training_profile(args, p, argv)
    normalize_data_selection(args, explicit)
    return args


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if not args.run_name:
        args.run_name = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    if args.resume_latest_best and not args.resume:
        args.resume = str(latest_best_model_path())
        print(f"  Reinforcement resume: {args.resume}")
    seed_everything(args.seed)
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if METRICS_FILE.exists():
        METRICS_FILE.unlink()

    print(f"\n{'═'*70}")
    print(f"  ADVANCED BUG BOUNTY MODEL — LIVE TRAINING PIPELINE")
    print(f"  Device: {DEVICE.upper()}", end="")
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f" | {props.name} | {props.total_memory/1e9:.1f} GB")
    else:
        print()
    print(f"{'═'*70}")
    print(f"  Project: {PROJECT_DIR}")
    print(f"  Raw data: {RAW_DIR}")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Logs: {LOGS_DIR}")
    if args.print_options:
        print_available_training_profiles()
    print_run_options(args)
    if args.dry_run:
        print("[TRAIN] dry run complete; no data collection or training started")
        return

    require_cuda_if_requested(args.require_cuda)

    if args.watch:
        start_dashboard(args.port)
        time.sleep(1)

    # ── Step 1: Collect live data ──────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  STEP 1/4: DATA COLLECTION")
    print(f"{'─'*70}")

    if args.skip_collect:
        print("  Using cached data; live collection skipped.")
    else:
        collect_all(
            nvd_per_kw=args.nvd_per_kw,
            gh_max=args.gh_max,
            osv_per_eco=args.osv_per_eco,
            h1_pages=args.h1_pages,
            synth_n=args.synth_n,
            github_token=args.github_token,
            nvd_api_key=args.nvd_api_key,
            skip_live=False,
        )
    enforce_min_data_gb(args.min_data_gb)
    require_data_files(args.require_data_file)

    # ── Step 2: Build datasets ────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  STEP 2/4: BUILDING DATASETS")
    print(f"{'─'*70}")

    tokenizer    = SimpleTokenizer(vocab_size=8000, max_len=256)
    text_records = load_all_records(bulk_percent=args.bulk_text_percent)
    http_records = load_http_records(bulk_percent=args.bulk_http_percent)

    if len(text_records) < 100:
        if args.no_synthetic_fallback:
            raise ValueError(
                f"Only {len(text_records)} usable text records found in {RAW_DIR}. "
                "--no-synthetic-fallback is set, so training will not use generated data."
            )
        print("  ⚠ Very few records — regenerating synthetic data")
        from live_data_collector import generate_advanced_http
        http_records = generate_advanced_http(n_samples=args.synth_n)
        # Create minimal text records from HTTP
        text_records = [{"text": f"HTTP attack: {r['label']}", "label": r["label"],
                         "severity": r.get("severity","medium"), "cvss_score": 5.0}
                        for r in http_records]

    write_dataset_manifest(text_records, http_records, args)

    tr_loader, vl_loader, te_loader = build_dataloaders(
        text_records, http_records, tokenizer, args
    )

    # ── Step 3: Build / load model ────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  STEP 3/4: MODEL SETUP")
    print(f"{'─'*70}")

    model = create_advanced_model(checkpoint_path=args.resume)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}"
          f"  Size: {model.get_model_size_mb():.1f} MB")

    # Save tokenizer config
    tok_config = {"vocab_size": 8000, "max_len": 256,
                  "label_map": LABEL_TO_IDX, "chain_map": CHAIN_TO_IDX,
                  "severity_map": SEV_TO_IDX}
    (MODELS_DIR / "tokenizer_config.json").write_text(json.dumps(tok_config, indent=2))

    # ── Step 4: Train ─────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  STEP 4/4: TRAINING")
    print(f"{'─'*70}")

    t_start = time.time()
    trainer = AdvancedTrainer(model, args)
    results = trainer.fit(tr_loader, vl_loader, te_loader)

    elapsed = time.time() - t_start
    print(f"\n{'═'*70}")
    print(f"  ✅ COMPLETE — {elapsed/60:.1f} min")
    print(f"  Best val F1:    {trainer.best_val_f1:.4f}")
    if results["test"]:
        tr = results["test"]
        print(f"  Test accuracy:  {tr['acc']:.4f}")
        print(f"  Test F1 (wtd):  {tr['f1_weighted']:.4f}")
        print(f"  Test F1 (mac):  {tr['f1_macro']:.4f}")
    print(f"  Models → {MODELS_DIR}")
    print(f"  Metrics→ {METRICS_FILE}")
    if args.watch:
        print(f"  Dashboard still live at http://localhost:{args.port}")
        print(f"  Press Ctrl+C to exit.")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            pass
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
