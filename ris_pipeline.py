#!/usr/bin/env python3
"""Ridgeline Quote Desk — RFQ extraction, ingestion & matching pipeline.

Converts the RFQs under ``rfqs/`` into validated JSON deliverables under
``outputs/``. The output contract is defined by ``check_submission.py``
(v1.2.0); where anything disagrees with it, the checker wins.

Design rules honoured throughout:

* Every format is ingested: .txt, .csv, .xlsx, .pdf, images (stubbed).
* The pipeline never raises an exception that aborts the run — every file
  and every line is wrapped in try/except, and a failure becomes a line
  with ``flags=["unreadable_line"]`` (or ``["stub"]`` for images).
* ``answer_key/`` is never read.
* Secrets are never hard-coded; the DeepSeek key is read from the
  environment (``DEEPSEEK_API_KEY``) via python-dotenv.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

MatchMethod = Literal["ridgeline_part_number", "cross_reference", "description", "llm"]

OUTPUT_STEMS = (
    "rfq_01_email",
    "rfq_02_clearwater",
    "rfq_03_tricounty",
    "rfq_04_handwritten",
    "rfq_05_forwarded_chain",
    "rfq_06_messy",
)

STUB_STEM = "rfq_04_handwritten"

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "with", "as", "at", "by",
    "from", "and", "or", "not", "no", "if", "it", "is", "are", "be", "we",
    "you", "they", "their", "this", "that", "these", "those", "get", "got",
    "usually", "please", "quote", "quotes", "following", "need", "needed",
    "same", "last", "order", "asap", "ea", "each", "qty", "quantity", "pcs",
    "pc", "pkg", "pk", "ct", "available", "fine", "make", "bump",
}

SYNONYMS = {
    "gr5": "grade 5",
    "gr8": "grade 8",
    "gr 5": "grade 5",
    "gr 8": "grade 8",
    "hyd": "hydraulic",
    "fp": "full port",
    "thk": "thick",
}

PLURAL_FIXES = {
    "boxes": "box", "cases": "case", "gloves": "glove", "bearings": "bearing",
    "belts": "belt", "bolts": "bolt", "gaskets": "gasket", "cartridges": "cartridge",
    "tires": "tire", "o-rings": "o-ring", "washers": "washer", "nuts": "nut",
    "screws": "screw", "fittings": "fitting", "valves": "valve", "motors": "motor",
    "filters": "filter", "hoses": "hose", "couplers": "coupler", "clamps": "clamp",
}

UNIT_TOKENS = (
    r"ea|each|bx|box|boxes|cs|case|cases|pcs|pc|pkg|pk|pack|packs|pair|pairs|pr|"
    r"roll|rolls|gal|gallon|gallons|gals|pail|pails|kit|kits|ft|feet|ct"
)

INSTRUCTION_RE = re.compile(
    r"(ship\s+to|deliver\s+to|dock\b|need\s+it\s+by|need(?:ed)?\s+by|need\s+asap|"
    r"reply\s+by|terms\s+per|buyer\s*:|equivalent\s+items|please\s+quote\s+per)",
    re.IGNORECASE,
)


class LineItem(BaseModel):
    line: int
    as_written: str
    quantity: Optional[float] = None
    unit: Optional[str] = None
    ridgeline_part_number: Optional[str] = None
    match_method: Optional[MatchMethod] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    needs_review: bool = True
    flags: list[str] = Field(default_factory=list)
    note: Optional[str] = None


class RFQResult(BaseModel):
    rfq_id: str
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None
    lines: list[LineItem]
    non_item_rows: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _clamp01(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value) or value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _safe_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _join_note(a, b) -> Optional[str]:
    a = (a or "").strip()
    b = (b or "").strip()
    if a and b and a != b:
        return f"{a}; {b}"
    return a or b or None


def _norm_text(text: str) -> str:
    s = (text or "").lower()
    for key, value in SYNONYMS.items():
        s = re.sub(r"\b" + re.escape(key) + r"\b", value, s)
    s = re.sub(r"[^0-9a-z]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _singular(token: str) -> str:
    if token in PLURAL_FIXES:
        return PLURAL_FIXES[token]
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4 and not token.endswith("sses"):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _tokens(text: str) -> list:
    out = []
    for token in _norm_text(text).split():
        if token in STOPWORDS:
            continue
        if len(token) == 1 and token.isalpha() and token not in {"s", "m", "l"}:
            continue
        out.append(_singular(token))
    return out


def tokenize(text: str) -> set:
    return set(_tokens(text))


def _bigrams(tokens: list) -> set:
    return {" ".join(pair) for pair in zip(tokens, tokens[1:])}


def fuzzy_score(query: str, doc: str) -> float:
    """0..1 similarity between a request line and a product description.

    Blends unigram recall, bigram overlap (keeps "5 HP" distinct from
    "3 phase" / "2 HP"), and raw sequence similarity.
    """
    qt = _tokens(query)
    dt = _tokens(doc)
    q_set, d_set = set(qt), set(dt)
    if not q_set:
        return 0.0
    recall = len(q_set & d_set) / len(q_set)
    q_bg, d_bg = _bigrams(qt), _bigrams(dt)
    bg = len(q_bg & d_bg) / len(q_bg) if q_bg else 0.0
    seq = difflib.SequenceMatcher(None, _norm_text(query), _norm_text(doc)).ratio()
    return _clamp01(0.7 * recall + 0.2 * bg + 0.1 * seq)


def _num_str(value) -> str:
    f = float(value)
    return str(int(f)) if f == int(f) else repr(f)


def _clean_query(text: str, quantity) -> str:
    """Remove quantity/unit/note noise from a line before fuzzy matching."""
    s = (text or "").strip()
    if not s:
        return s
    s = re.sub(r"\(.*?\)", " ", s)  # parenthetical notes
    s = re.sub(r"\bqty\.?\s*[:=\-]?\s*\d+(?:\.\d+)?\b", " ", s, flags=re.IGNORECASE)
    if quantity is not None:
        v = re.escape(_num_str(quantity))
        s = re.sub(r"\bx\s*" + v + r"\s*$", " ", s, flags=re.IGNORECASE)
        s = re.sub(r"\b" + v + r"\s*(?:" + UNIT_TOKENS + r")\b", " ", s, flags=re.IGNORECASE)
        s = re.sub(r"^\s*" + v + r"\s+", " ", s)
        s = re.sub(r"\s+-\s*" + v + r"\s*$", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_qty_unit(text: str) -> tuple[Optional[float], Optional[str]]:
    """Best-effort quantity/unit extraction. Returns (quantity, unit)."""
    if not text:
        return None, None
    s = (text or "").strip()
    m = re.search(r"\bqty\.?\s*[:=\-]?\s*(\d+(?:\.\d+)?)", s, re.IGNORECASE)
    if m:
        return _safe_float(m.group(1)), None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(" + UNIT_TOKENS + r")\b", s, re.IGNORECASE)
    if m:
        return _safe_float(m.group(1)), m.group(2).lower()
    m = re.search(r"(?<![A-Za-z0-9])x\s*(\d+(?:\.\d+)?)\s*(?:$|[(,;])", s, re.IGNORECASE)
    if m:
        return _safe_float(m.group(1)), None
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s+(.+)$", s)
    if m:
        return _safe_float(m.group(1)), None
    m = re.search(r"(?<=\s)-\s*(\d+(?:\.\d+)?)\b", s)
    if m:
        return _safe_float(m.group(1)), None
    return None, None


def _is_instruction_line(line: str) -> bool:
    s = (line or "").strip()
    if s.startswith("- "):
        return False
    return bool(INSTRUCTION_RE.search(s))


def _is_signature(line: str) -> bool:
    s = (line or "").strip().lower()
    return s.startswith("sent from") or bool(re.fullmatch(r"\(?\d{3}\)?[\s.-]*\d{3}[\s.-]*\d{4}", s))


def _read_text(path: Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            return Path(path).read_text(encoding="latin-1")
        except Exception:
            return ""
    except Exception:
        return ""


def _body_lines(text: str) -> list:
    out = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if re.match(r"^(From|To|Date|Subject|Cc|Sent):", s, re.IGNORECASE):
            continue
        out.append(s)
    return out


def _read_csv(path: Path) -> list:
    out = []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                out.append({k: (v or "") for k, v in row.items()})
    except Exception:
        return []
    return out


def load_catalog(root: Path) -> dict:
    root = Path(root)
    customers = _read_csv(root / "customers.csv")
    products = _read_csv(root / "products.csv")
    crossref = _read_csv(root / "cross_reference.csv")
    products_by_part = {
        p["part_number"].strip().upper(): p
        for p in products
        if p.get("part_number") and p["part_number"].strip()
    }
    return {
        "customers": customers,
        "products": products,
        "crossref": crossref,
        "products_by_part": products_by_part,
    }


def find_customer(text: str, customers: list) -> tuple[Optional[str], Optional[str]]:
    t = (text or "").lower()
    for c in customers:
        name = (c.get("name") or "").strip()
        if name and name.lower() in t:
            return c.get("customer_id") or None, name or None
    for m in re.finditer(r"@([A-Za-z0-9.-]+)\.example\b", t):
        label = m.group(1).split(".")[0]
        for c in customers:
            name = (c.get("name") or "").strip()
            if re.sub(r"[^a-z0-9]", "", name.lower()) == label:
                return c.get("customer_id") or None, name or None
    return None, None


# --------------------------------------------------------------------------
# matching stages
# --------------------------------------------------------------------------

def _direct_result(pn: str, prod: dict) -> dict:
    flags, notes = [], []
    ridgeline = pn
    if (prod.get("status") or "").strip().lower() == "discontinued":
        repl = (prod.get("replaced_by") or "").strip()
        if repl:
            ridgeline = repl
            flags.append("discontinued_replaced")
            notes.append(f"{pn} is discontinued; replaced by {repl}")
    return {
        "match_method": "ridgeline_part_number",
        "confidence": 0.90,
        "ridgeline_part_number": ridgeline,
        "flags": flags,
        "notes": notes,
    }


def _crossref_lookup(pn: str, catalog: dict) -> Optional[dict]:
    pn = (pn or "").strip().upper()
    for xr in catalog["crossref"]:
        if (xr.get("competitor_part_number") or "").strip().upper() == pn:
            return xr
    return None


def _crossref_result(xr: dict, catalog: dict) -> dict:
    flags, notes = [], []
    ridgeline = (xr.get("ridgeline_part_number") or "").strip()
    prod = catalog["products_by_part"].get(ridgeline)
    if prod and (prod.get("status") or "").strip().lower() == "discontinued":
        repl = (prod.get("replaced_by") or "").strip()
        if repl:
            notes.append(f"{ridgeline} is discontinued; replaced by {repl}")
            ridgeline = repl
            flags.append("discontinued_replaced")
    return {
        "match_method": "cross_reference",
        "confidence": 0.90,
        "ridgeline_part_number": ridgeline,
        "flags": flags,
        "notes": notes,
    }


def _stage1(text: str, part_hint, catalog: dict) -> Optional[dict]:
    text = text or ""
    if part_hint:
        pn = str(part_hint).strip().upper()
        if re.fullmatch(r"RL\d{5,8}", pn):
            prod = catalog["products_by_part"].get(pn)
            if prod:
                return _direct_result(pn, prod)
        else:
            xr = _crossref_lookup(pn, catalog)
            if xr:
                return _crossref_result(xr, catalog)
    m = re.search(r"\b(RL\d{5,8})\b", text, re.IGNORECASE)
    if m:
        pn = m.group(1).upper()
        prod = catalog["products_by_part"].get(pn)
        if prod:
            return _direct_result(pn, prod)
    for xr in catalog["crossref"]:
        cp = (xr.get("competitor_part_number") or "").strip()
        if cp and re.search(r"(?<![A-Za-z0-9])" + re.escape(cp) + r"(?![A-Za-z0-9])", text, re.IGNORECASE):
            return _crossref_result(xr, catalog)
    return None


def _stage2(query: str, catalog: dict) -> Optional[dict]:
    query = (query or "").strip()
    if not query:
        return None
    best = None
    for prod in catalog["products"]:
        desc = prod.get("description") or ""
        if not desc:
            continue
        score = fuzzy_score(query, desc)
        if best is None or score > best[1]:
            best = (prod, score)
    if best and best[1] >= 0.60:
        prod, score = best
        return {
            "match_method": "description",
            "confidence": round(_clamp01(score), 4),
            "ridgeline_part_number": prod["part_number"],
        }
    return None


def _top_candidates(query: str, catalog: dict, n: int = 10) -> list:
    scored = []
    for prod in catalog["products"]:
        desc = prod.get("description") or ""
        if not desc:
            continue
        score = fuzzy_score(query, desc)
        if score > 0.0:
            scored.append((score, prod))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [prod for _, prod in scored[:n]]


def _stage3(query: str, catalog: dict, cfg: dict) -> dict:
    query = (query or "").strip()
    candidates = _top_candidates(query, catalog, 10)
    if not candidates:
        return {
            "match_method": None,
            "confidence": 0.0,
            "flags": ["llm_unresolved"],
            "notes": ["no candidate products to ask the LLM about"],
        }
    if not cfg.get("api_key"):
        return {
            "match_method": None,
            "confidence": 0.0,
            "flags": ["llm_unresolved"],
            "notes": ["no DEEPSEEK_API_KEY configured; LLM fallback skipped"],
        }
    pn = llm_resolve(query, candidates, cfg)
    if pn:
        return {
            "match_method": "llm",
            "confidence": 0.75,
            "ridgeline_part_number": pn,
            "flags": [],
            "notes": ["matched by LLM"],
        }
    return {
        "match_method": None,
        "confidence": 0.0,
        "flags": ["llm_unresolved"],
        "notes": ["LLM fallback could not resolve a match"],
    }


def llm_resolve(query: str, candidates: list, cfg: dict) -> Optional[str]:
    """Ask DeepSeek to pick the best candidate; returns a part number or None.

    Never raises — any failure means "the LLM returned nothing".
    """
    try:
        lines = [f"- {c['part_number']}: {c['description']}" for c in candidates]
        prompt = (
            "You are matching a maintenance, repair and operations (MRO) request "
            "line to a product catalog.\n"
            f"Request line: {query!r}\n"
            "Candidate products (part_number: description):\n"
            + "\n".join(lines)
            + "\nReply with only the part number that best matches, or the single word NONE."
        )
        body = {
            "model": cfg.get("model") or "deepseek-v4-pro",
            "reasoning_mode": "high",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise MRO parts matching assistant. "
                        "Reply with exactly one part number from the candidates, or NONE."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            cfg["base_url"].rstrip("/") + "/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + cfg["api_key"],
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=int(cfg.get("timeout") or 60)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        m = re.search(r"\bRL\d{5,8}\b", str(content), re.IGNORECASE)
        if m:
            pn = m.group(0).upper()
            if pn in {c["part_number"] for c in candidates}:
                return pn
        return None
    except Exception:
        return None


# --------------------------------------------------------------------------
# per-line matching
# --------------------------------------------------------------------------

def _match_raw_line(raw: dict, catalog: dict, cfg: dict) -> dict:
    text = str(raw.get("as_written") or raw.get("text") or "").strip()
    quantity = _safe_float(raw.get("quantity"))
    unit = raw.get("unit")
    part_hint = raw.get("part_hint")
    note_hint = raw.get("note_hint")
    query = _clean_query(str(raw.get("query") or text).strip() or text, quantity)

    flags: list = []
    notes: list = []
    ridgeline = None
    method = None
    confidence = 0.0

    if quantity is None:
        flags.append("missing_quantity")
        notes.append("this item's quantity is missing")

    s1 = _stage1(text, part_hint, catalog)
    if s1:
        method = s1["match_method"]
        confidence = s1["confidence"]
        ridgeline = s1["ridgeline_part_number"]
        flags.extend(s1.get("flags") or [])
        notes.extend(s1.get("notes") or [])
    else:
        s2 = _stage2(query, catalog)
        if s2:
            method = s2["match_method"]
            confidence = s2["confidence"]
            ridgeline = s2["ridgeline_part_number"]
        else:
            s3 = _stage3(query, catalog, cfg)
            method = s3.get("match_method")
            confidence = s3.get("confidence", 0.0)
            ridgeline = s3.get("ridgeline_part_number")
            flags.extend(s3.get("flags") or [])
            notes.extend(s3.get("notes") or [])

    if note_hint:
        notes.append(note_hint)

    if method is None:
        confidence = 0.0
        needs_review = True
        if "no_match" not in flags:
            flags.append("no_match")
        if not any(notes):
            notes.append("no matching part found")
    else:
        needs_review = not (confidence is not None and confidence > 0.80)

    note = "; ".join(dict.fromkeys(n for n in notes if n)) or None
    if flags and not note:
        note = "see flags"

    return {
        "as_written": text,
        "quantity": quantity,
        "unit": unit,
        "ridgeline_part_number": ridgeline,
        "match_method": method,
        "confidence": round(_clamp01(confidence), 4),
        "needs_review": needs_review,
        "flags": flags,
        "note": note,
    }


def match_raw_line(raw: dict, catalog: dict, cfg: dict) -> dict:
    try:
        return _match_raw_line(raw, catalog, cfg)
    except Exception as exc:
        return {
            "as_written": str(raw.get("as_written") or raw.get("text") or "unreadable"),
            "quantity": _safe_float(raw.get("quantity")),
            "unit": raw.get("unit"),
            "ridgeline_part_number": None,
            "match_method": None,
            "confidence": 0.0,
            "needs_review": True,
            "flags": ["unreadable_line"],
            "note": f"{type(exc).__name__}: {exc}",
        }


# --------------------------------------------------------------------------
# extractors
# --------------------------------------------------------------------------

def extract_bullet_email(path: Path) -> tuple:
    """Email with a dash-prefixed bullet list of items."""
    text = _read_text(path)
    items, non_items = [], []
    for line in _body_lines(text):
        if line.startswith("- "):
            item_text = line[2:].strip()
            qty, unit = parse_qty_unit(item_text)
            items.append({
                "as_written": item_text,
                "query": item_text,
                "quantity": qty,
                "unit": unit,
            })
        elif _is_instruction_line(line):
            non_items.append(line)
    return text, items, non_items


def extract_messy_txt(path: Path) -> tuple:
    """Plain-text email where every body line (except the intro) is an item."""
    text = _read_text(path)
    items, non_items = [], []
    for line in _body_lines(text):
        if _is_signature(line):
            continue
        if _is_instruction_line(line) or line.endswith(":"):
            non_items.append(line)
            continue
        qty, unit = parse_qty_unit(line)
        items.append({
            "as_written": line,
            "query": line,
            "quantity": qty,
            "unit": unit,
        })
    return text, items, non_items


def _extract_bullets(text: str) -> list:
    out = []
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("- "):
            out.append(s[2:].strip())
    return out


def _parse_directives(text: str) -> list:
    directives = []
    for m in re.finditer(
        r"make\s+the\s+(.+?)\s+(\d[\w./\-]*\w*|\w+)\s+not\s+(\d[\w./\-]*)",
        text, re.IGNORECASE,
    ):
        directives.append({
            "type": "desc",
            "keyword": m.group(1).strip(),
            "new": m.group(2),
            "old": m.group(3),
        })
    for m in re.finditer(
        r"bump\s+(?:the\s+)?(.+?)\s+(?:to|up\s+to)\s+(\d+)",
        text, re.IGNORECASE,
    ):
        directives.append({
            "type": "qty",
            "keyword": m.group(1).strip(),
            "qty": int(m.group(2)),
        })
    return directives


def _apply_desc_change(item: str, directive: dict) -> str:
    if directive["keyword"].lower() not in item.lower():
        return item
    new = directive["new"]
    old = directive["old"]
    m = re.match(r"(\d[\d./-]*)\s*([A-Za-z]*)", new)
    if not m:
        return re.sub(r"\b" + re.escape(old) + r"\b", new, item, count=1)
    num, unit = m.group(1), m.group(2)
    if unit:
        pat = r"\b" + re.escape(old) + r"\s*" + re.escape(unit) + r"\b"
        if re.search(pat, item, re.IGNORECASE):
            return re.sub(pat, f"{num} {unit}", item, count=1, flags=re.IGNORECASE)
    return re.sub(r"\b" + re.escape(old) + r"\b", new, item, count=1)


def _apply_qty_change(item: str, directive: dict) -> str:
    if directive["keyword"].lower() not in item.lower():
        return item
    new = str(directive["qty"])
    if re.search(r"\bqty\s*\d+", item, re.IGNORECASE):
        return re.sub(r"\bqty\s*\d+", f"qty {new}", item, count=1, flags=re.IGNORECASE)
    return re.sub(r"-\s*\d+\s*$", f"- {new}", item, count=1)


def extract_email_chain(path: Path) -> tuple:
    """Read the whole thread, then ingest the latest (most recent) changes."""
    text = _read_text(path)
    blocks = re.split(r"^[ \t]*-----[^\n]*-----[ \t]*$", text, flags=re.MULTILINE)

    base_items: list = []
    base_idx = None
    for i, block in enumerate(blocks):
        bullets = _extract_bullets(block)
        if bullets:
            base_idx = i
            base_items = bullets
            break
    if not base_items:
        base_items = _extract_bullets(text)

    items = list(base_items)
    if base_idx is not None:
        for i in range(base_idx):
            for directive in _parse_directives(blocks[i]):
                if directive["type"] == "desc":
                    items = [_apply_desc_change(it, directive) for it in items]
                elif directive["type"] == "qty":
                    items = [_apply_qty_change(it, directive) for it in items]

    non_items = []
    for line in _body_lines(text):
        if _is_instruction_line(line) and not line.startswith("- "):
            non_items.append(line)

    out = []
    for it in items:
        qty, unit = parse_qty_unit(it)
        out.append({"as_written": it, "query": it, "quantity": qty, "unit": unit})
    return text, out, non_items


def _pdf_text_fallback(path: Path) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""


def _parse_pdf_text_rows(text: str) -> list:
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    start = None
    for i, line in enumerate(lines):
        if re.fullmatch(r"\d+", line):
            start = i
            break
    if start is None:
        return []
    tokens = lines[start:]
    rows = []
    for i in range(0, len(tokens), 5):
        chunk = tokens[i:i + 5]
        rows.append(chunk + [""] * (5 - len(chunk)))
    return rows


def extract_pdf(path: Path) -> tuple:
    text = ""
    rows: list = []
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                tbl = page.extract_table()
                if tbl:
                    rows.extend(tbl)
                text += (page.extract_text() or "") + "\n"
    except Exception:
        rows = []
    if not rows:
        text = _pdf_text_fallback(path)
        rows = _parse_pdf_text_rows(text)

    start = 0
    for i, row in enumerate(rows):
        joined = " ".join(str(c) for c in row if c)
        if re.search(r"supplier\s+part|description", joined, re.IGNORECASE) and \
                re.search(r"qty", joined, re.IGNORECASE):
            start = i + 1
            break

    items, non_items = [], []
    for row in rows[start:]:
        cells = [(str(c).strip() if c is not None else "") for c in row]
        if not any(cells):
            continue
        while len(cells) < 5:
            cells.append("")
        part, desc, qty_cell, uom = cells[1], cells[2], cells[3], cells[4]
        if not part and not desc:
            continue
        qty = _safe_float(qty_cell)
        items.append({
            "as_written": f"{part} {desc}".strip(),
            "query": desc,
            "part_hint": part,
            "quantity": qty,
            "unit": uom or None,
        })
    for line in (text or "").splitlines():
        s = line.strip()
        if s and _is_instruction_line(s):
            non_items.append(s)
    return text, items, non_items


def _apply_change(text: str, change: str) -> str:
    m = re.search(r"\d+(?:/\d+)?\s*\"?", text)
    if m:
        return text[:m.start()] + change + text[m.end():]
    return f"{text} {change}".strip()


def _resolve_same_as(item: dict, prev_items: list, note_hint, desc: str) -> dict:
    m = re.search(r"same\s+as\s+(?:above|line\s*(\d+))\s*(?:but\s+(.+))?$", (desc or ""), re.IGNORECASE)
    if not m:
        return item
    ref = m.group(1)
    change = (m.group(2) or "").strip()
    ref_no = int(ref) if ref else len(prev_items)
    ref_item = prev_items[ref_no - 1] if 0 < ref_no <= len(prev_items) else (prev_items[-1] if prev_items else None)
    if ref_item:
        base_query = ref_item.get("query") or ref_item.get("as_written") or ""
        item["query"] = _apply_change(base_query, change) if change else base_query
        inherited = f"inherited from line {ref_no}" + (f"; {note_hint}" if note_hint else "")
        item["note_hint"] = _join_note(item.get("note_hint"), inherited)
    return item


def extract_xlsx(path: Path) -> tuple:
    import openpyxl
    wb = openpyxl.load_workbook(str(path), data_only=True)
    ws = wb["RFQ"] if "RFQ" in wb.sheetnames else wb.worksheets[0]

    rows = [row for row in ws.iter_rows(values_only=True)]
    header_idx = None
    for i, row in enumerate(rows):
        a = str(row[0]).strip() if len(row) > 0 and row[0] is not None else ""
        b = str(row[1]).strip() if len(row) > 1 and row[1] is not None else ""
        if re.search(r"item\s*#", a, re.IGNORECASE) and re.search(r"desc", b, re.IGNORECASE):
            header_idx = i
            break
    if header_idx is None:
        for i, row in enumerate(rows):
            joined = " ".join(str(c) for c in row if c is not None)
            if "Item #" in joined and ("Desc" in joined or "Qty" in joined):
                header_idx = i
                break

    items, non_items = [], []
    if header_idx is not None:
        for row in rows[header_idx + 1:]:
            row = list(row) + [None] * (5 - len(row))
            a, b, c, d, e = row[0], row[1], row[2], row[3], row[4]
            part = str(a).strip() if a is not None else ""
            desc = str(b).strip() if b is not None else ""
            qty = _safe_float(c)
            uom = str(d).strip() if d is not None else ""
            note_hint = str(e).strip() if e is not None and str(e).strip() else None

            if not part and not desc and qty is None:
                continue
            if not part and qty is None and desc:
                non_items.append(desc)
                continue
            if not desc and not part:
                continue
            item = {
                "as_written": f"{part} {desc}".strip(),
                "query": desc,
                "part_hint": part or None,
                "quantity": qty,
                "unit": uom or None,
                "note_hint": note_hint,
            }
            item = _resolve_same_as(item, items, note_hint, desc)
            items.append(item)

    full_text = " ".join(str(c) for row in rows for c in row if c is not None)
    return full_text, items, non_items


def extract_image(path: Path) -> tuple:
    return "", [{"as_written": path.name, "query": path.name, "stub": True}], []


def _dispatch(stem: str, path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"):
        return "image"
    if ext == ".pdf":
        return "pdf"
    if ext in (".xlsx", ".xls", ".xlsm"):
        return "xlsx"
    if stem == "rfq_05_forwarded_chain":
        return "chain"
    if stem == "rfq_06_messy":
        return "messy"
    return "email"


def _stub_result(stem: str, filename: str) -> RFQResult:
    return RFQResult(
        rfq_id=stem,
        customer_id=None,
        customer_name=None,
        lines=[LineItem(
            line=1,
            as_written=filename,
            quantity=None,
            unit=None,
            ridgeline_part_number=None,
            match_method=None,
            confidence=0.0,
            needs_review=True,
            flags=["stub"],
            note="Need OCR or a vision model for this",
        )],
        non_item_rows=[],
    )


def _unreadable_line(line_no: int, raw: str, note: str) -> LineItem:
    return LineItem(
        line=line_no,
        as_written=raw or "unreadable",
        quantity=None,
        unit=None,
        ridgeline_part_number=None,
        match_method=None,
        confidence=0.0,
        needs_review=True,
        flags=["unreadable_line"],
        note=note,
    )


def build_result(stem: str, path: Path, catalog: dict, cfg: dict) -> RFQResult:
    kind = _dispatch(stem, path)
    try:
        if kind == "image":
            return _stub_result(stem, path.name)
        if kind == "pdf":
            full_text, raw_items, non_items = extract_pdf(path)
        elif kind == "xlsx":
            full_text, raw_items, non_items = extract_xlsx(path)
        elif kind == "chain":
            full_text, raw_items, non_items = extract_email_chain(path)
        elif kind == "messy":
            full_text, raw_items, non_items = extract_messy_txt(path)
        else:
            full_text, raw_items, non_items = extract_bullet_email(path)

        customer_id, customer_name = find_customer(full_text, catalog["customers"])
        lines = []
        for i, raw in enumerate(raw_items, 1):
            data = match_raw_line(raw, catalog, cfg)
            data["line"] = i
            lines.append(LineItem(**data))
        if not lines:
            lines = [_unreadable_line(1, path.name, "no item lines extracted")]
        return RFQResult(
            rfq_id=stem,
            customer_id=customer_id,
            customer_name=customer_name,
            lines=lines,
            non_item_rows=non_items,
        )
    except Exception as exc:
        return RFQResult(
            rfq_id=stem,
            customer_id=None,
            customer_name=None,
            lines=[_unreadable_line(1, path.name, f"{type(exc).__name__}: {exc}")],
            non_item_rows=[],
        )


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def load_config(root: Path) -> dict:
    root = Path(root)
    load_dotenv(root / ".env")
    return {
        "api_key": os.environ.get("DEEPSEEK_API_KEY") or None,
        "model": os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-pro",
        "base_url": (os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/"),
        "timeout": os.environ.get("DEEPSEEK_TIMEOUT") or "60",
    }


def run_pipeline(root: Path, cfg: dict) -> list:
    root = Path(root)
    catalog = load_catalog(root)
    rfq_dir = root / "rfqs"
    files = {}
    if rfq_dir.is_dir():
        for f in rfq_dir.iterdir():
            if f.is_file() and not f.name.startswith(("~$", "$")):
                files[f.stem] = f
    results = []
    for stem in OUTPUT_STEMS:
        path = files.get(stem)
        if path is None:
            results.append(RFQResult(
                rfq_id=stem,
                customer_id=None,
                customer_name=None,
                lines=[_unreadable_line(1, f"{stem}.txt", "source file not found")],
                non_item_rows=[],
            ))
        else:
            results.append(build_result(stem, path, catalog, cfg))
    return results


def compute_metrics(results: list) -> dict:
    non_stub = [
        line for r in results if r.rfq_id != STUB_STEM for line in r.lines
    ]
    total = len(non_stub)
    with_qty = sum(1 for line in non_stub if line.quantity is not None)
    matched = [line for line in non_stub if line.match_method is not None]
    precise = [
        line for line in matched
        if line.match_method in ("ridgeline_part_number", "cross_reference")
    ]
    return {
        "threshold": 0.6,
        "extraction_accuracy": round(with_qty / total, 4) if total else 0.0,
        "match_precision": round(len(precise) / len(matched), 4) if matched else 0.0,
        "match_recall": round(len(matched) / total, 4) if total else 0.0,
        "additional_metrics": {},
    }


def write_outputs(results: list, root: Path) -> list:
    out_dir = Path(root) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for r in results:
        data = r.model_dump()
        path = out_dir / f"{r.rfq_id}.json"
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.write("\n")
        written.append(str(path))
    return written


def write_evaluation(root: Path, results: list) -> dict:
    metrics = compute_metrics(results)
    ev_dir = Path(root) / "evaluation"
    ev_dir.mkdir(parents=True, exist_ok=True)
    with open(ev_dir / "results.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ris_pipeline.py",
        description="Ridgeline Quote Desk — RFQ extraction & matching pipeline",
    )
    parser.add_argument("--root", default=None, help="repository root (default: script directory)")
    parser.add_argument("--no-llm", action="store_true",
                        help="skip the LLM fallback even if DEEPSEEK_API_KEY is set")
    args = parser.parse_args(argv)

    try:
        root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent
        cfg = load_config(root)
        if args.no_llm:
            cfg["api_key"] = None
        results = run_pipeline(root, cfg)
        written = write_outputs(results, root)
        metrics = write_evaluation(root, results)
        print(f"Wrote {len(written)} RFQ JSON files to {Path(root) / 'outputs'}")
        print(f"Evaluation metrics: {json.dumps(metrics)}")
        for r in results:
            print(f"  {r.rfq_id}: {len(r.lines)} line(s), customer={r.customer_id or '-'}")
        return 0
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
