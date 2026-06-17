"""
inference.py
============
Inference entrypoint for the advanced 34-class bug bounty model.

Examples:
  python inference.py --text "SSRF lets a user fetch http://169.254.169.254/"
  python inference.py --text-file report.txt
  python inference.py --request-json sample_http.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR.parent
MODELS_DIR = BASE_DIR / "models"
sys.path.insert(0, str(PROJECT_DIR))

from advanced_model import SimpleTokenizer, create_advanced_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _default_checkpoint() -> Path:
    for name in ("advanced_model_best.pt", "advanced_model_final.pt"):
        path = MODELS_DIR / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No advanced checkpoint found in {MODELS_DIR}. "
        "Run advanced_train.py first."
    )


def _http_to_text(request: dict, response: dict | None = None) -> str:
    response = response or {}
    return (
        f"HTTP {request.get('method', 'GET')} {request.get('path', '')} "
        f"headers={json.dumps(request.get('headers', {}), sort_keys=True)[:500]} "
        f"body={str(request.get('body', ''))[:2000]} "
        f"response_status={response.get('status_code', '')} "
        f"response={str(response.get('body') or response.get('body_snippet') or '')[:2000]}"
    )


def _http_features(request: dict, response: dict | None = None) -> list[float]:
    response = response or {}
    path = (request.get("path", "") or "").lower()
    body = (request.get("body", "") or "").lower()
    method = (request.get("method", "GET") or "GET").upper()
    resp_body = (response.get("body_snippet") or response.get("body") or "").lower()
    status = int(response.get("status_code", 200) or 200)
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
    return [float(v) for v in features[:64]]


class VulnerabilityDetector:
    """Small inference wrapper around the advanced multi-task model."""

    def __init__(self, checkpoint: str | Path | None = None):
        self.checkpoint = Path(checkpoint) if checkpoint else _default_checkpoint()
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")

        self.tokenizer = SimpleTokenizer(vocab_size=8000, max_len=256)
        self.model = create_advanced_model(str(self.checkpoint)).to(DEVICE)
        self.model.eval()

    def classify_text(self, text: str) -> dict:
        token_ids, attention_mask = self.tokenizer.batch_tokenize([text])
        token_ids = token_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        result = self.model.predict(token_ids=token_ids, attention_mask=attention_mask)
        result["url"] = ""
        result["checkpoint"] = str(self.checkpoint)
        result["device"] = DEVICE
        result["input_type"] = "text"
        return result

    def classify_http(self, request: dict, response: dict | None = None) -> dict:
        text = _http_to_text(request, response)
        token_ids, attention_mask = self.tokenizer.batch_tokenize([text])
        features = torch.tensor([_http_features(request, response)], dtype=torch.float, device=DEVICE)
        feature_mask = torch.tensor([True], dtype=torch.bool, device=DEVICE)
        result = self.model.predict(
            token_ids=token_ids.to(DEVICE),
            attention_mask=attention_mask.to(DEVICE),
            http_features=features,
            http_feature_mask=feature_mask,
        )
        result["url"] = str(request.get("url") or request.get("path") or "")
        result["checkpoint"] = str(self.checkpoint)
        result["device"] = DEVICE
        result["input_type"] = "http"
        return result


def _load_http_payload(path: Path) -> tuple[dict, dict | None]:
    payload = json.loads(path.read_text())
    if "request" in payload:
        return payload.get("request") or {}, payload.get("response")
    return payload, None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with the advanced bug bounty model."
    )
    parser.add_argument("--text", type=str, default=None, help="Text to classify")
    parser.add_argument("--text-file", type=Path, default=None, help="File containing report text")
    parser.add_argument("--request-json", type=Path, default=None, help="HTTP sample JSON file")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path")
    parser.add_argument("--compact", action="store_true", help="Print compact JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    detector = VulnerabilityDetector(args.checkpoint)

    if args.text is not None:
        result = detector.classify_text(args.text)
    elif args.text_file is not None:
        result = detector.classify_text(args.text_file.read_text())
    elif args.request_json is not None:
        request, response = _load_http_payload(args.request_json)
        result = detector.classify_http(request, response)
    elif not sys.stdin.isatty():
        result = detector.classify_text(sys.stdin.read())
    else:
        raise SystemExit("Provide --text, --text-file, --request-json, or stdin text.")

    print(json.dumps(result, indent=None if args.compact else 2))


if __name__ == "__main__":
    main()
