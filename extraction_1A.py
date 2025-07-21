# Imports
import os
import fitz  # PyMuPDF
import json
import numpy as np
import re
import bisect
import itertools
from collections import defaultdict
from thefuzz import fuzz
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

# Class Definition
class SmartPDFOutline:
    def __init__(self, pdf_path, ocr_dpi=300):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.lines = []
        self.features = {}
        self.elements = []
        self.ocr_dpi = ocr_dpi
        self.semantic_heading_patterns = re.compile(
            r'^(abstract|introduction|summary|background|overview|conclusion|references|appendix|contents|figure|table|chapter|section|part|phase|postscript|preamble|glossary|index|acknowledgements|methodology|results|discussion|evaluation|approach|requirements|milestones|timeline|membership|terms of reference)\b|'
            r'^\d+(\.\d+)*\s+[A-Z]',
            re.IGNORECASE
        )

    def _needs_ocr(self):
        total_area = 0
        text_area = 0
        for page in self.doc:
            total_area += abs(page.rect)
            for b in page.get_text("blocks"):
                r = fitz.Rect(b[:4])
                text_area += abs(r)
        return (text_area / total_area) < 0.01

    def _ocr_page(self, pnum):
        images = convert_from_path(self.pdf_path, dpi=self.ocr_dpi,
                                   first_page=pnum+1, last_page=pnum+1)
        if not images:
            return ""
        return pytesseract.image_to_string(images[0])

    def _is_semantic_heading(self, text):
        return bool(self.semantic_heading_patterns.match(text.strip()) and len(text.split()) < 12)

    def _is_visual_heading(self, line):
        sizes = [l["size"] for l in self.lines]
        if not sizes:
            return False
        threshold = np.percentile(sizes, 95)
        return line["size"] >= threshold

    def _build_lines_with_features(self):
        for pnum, page in enumerate(self.doc):
            if self._needs_ocr():
                ocr_text = self._ocr_page(pnum)
                y = 0
                for line in ocr_text.splitlines():
                    if not line.strip():
                        continue
                    bbox = fitz.Rect(0, y*12, page.rect.width, y*12 + 12)
                    self.lines.append({
                        "text": line.strip(),
                        "size": 12,
                        "bold": False,
                        "page": pnum,
                        "bbox": bbox,
                        "centeredness": 0.5,
                        "is_semantic": False
                    })
                    y += 1
                continue

            page_width = page.rect.width
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_LIGATURES)["blocks"]
            for block in blocks:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    spans = [s for s in line["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    bbox = fitz.Rect(line["bbox"])
                    x_center = (bbox.x0 + bbox.x1) / 2
                    centeredness = max(0, 1 - abs((page_width / 2) - x_center) / (page_width / 2))
                    avg_size = sum(s["size"] for s in spans) / len(spans)
                    is_bold = any(s["flags"] & 16 for s in spans)
                    text = " ".join(s["text"] for s in spans).strip()
                    is_sem = self._is_semantic_heading(text)
                    self.lines.append({
                        "text": text,
                        "size": avg_size,
                        "bold": is_bold,
                        "page": pnum,
                        "bbox": bbox,
                        "centeredness": centeredness,
                        "is_semantic": is_sem
                    })

    def _identify_headers_and_footers(self, top_n=5, bottom_n=3, threshold=90, tol=5):
        pages = defaultdict(list)
        for ln in self.lines:
            pages[ln["page"]].append(ln)

        sorted_pages = {}
        for pnum, lines in pages.items():
            lines.sort(key=lambda l: l["bbox"].y0)
            sorted_pages[pnum] = {
                "lines": lines,
                "y_coords": [l["bbox"].y0 for l in lines]
            }

        candidates = set()
        for pnum in range(1, len(self.doc)):
            prev = sorted_pages.get(pnum-1)
            curr = sorted_pages.get(pnum)
            if not prev or not curr:
                continue
            def match(curr_lines):
                for ln in curr_lines:
                    y = ln["bbox"].y0
                    lo = bisect.bisect_left(prev["y_coords"], y - tol)
                    hi = bisect.bisect_right(prev["y_coords"], y + tol)
                    for i in range(lo, hi):
                        pl = prev["lines"][i]
                        if fuzz.ratio(ln["text"], pl["text"]) > threshold:
                            candidates.add((ln["text"], int(y)))
                            candidates.add((pl["text"], int(pl["bbox"].y0)))
                            break
            match(curr["lines"][:top_n])
            match(curr["lines"][-bottom_n:])
        return candidates

    def _compute_statistical_features(self):
        sizes = [l["size"] for l in self.lines if l["size"] < 18]
        arr = np.array(sizes) if sizes else np.array([12])
        self.features = {
            "mean_size": float(arr.mean()),
            "std_dev_size": float(arr.std()),
            "median_size": float(np.median(arr))
        }

    def _classify_lines_as_headings(self):
        m, sd, md = (self.features[k] for k in ("mean_size", "std_dev_size", "median_size"))
        for l in self.lines:
            is_head = (
                self._is_semantic_heading(l["text"]) or
                self._is_visual_heading(l)
            ) and (l["bold"] or l["size"] > m + 1.5 * sd)
            if is_head and len(l["text"].split()) <= 20:
                self.elements.append({**l, "cls": "HEADING"})

    def _assemble_final_outline(self):
        page0 = [l for l in self.lines if l["page"] == 0]
        cand = sorted(page0, key=lambda l: (l["size"], len(l["text"])), reverse=True)
        title = next((c["text"] for c in cand if len(c["text"]) > 5), "Untitled Document")
        title_set = {c["text"] for c in cand[:2]}
        heads = [e for e in self.elements if e["text"] not in title_set]

        uniq = sorted({h["size"] for h in heads}, reverse=True)
        size2lvl = {sz: f"H{idx+1}" for idx, sz in enumerate(uniq)}
        outline = []
        for h in heads:
            lvl = size2lvl.get(h["size"], "H4")
            outline.append({"level": lvl, "text": h["text"], "page": h["page"]})

        order = {"H1": 1, "H2": 2, "H3": 3, "H4": 4}
        outline.sort(key=lambda x: (x["page"], order.get(x["level"], 99)))
        final = [next(g) for _, g in itertools.groupby(outline, key=lambda x: (x["text"], x["page"]))]
        return {"title": title, "outline": final}

    def analyze(self):
        self._build_lines_with_features()
        lines_by_page = defaultdict(list)
        for l in self.lines:
            lines_by_page[l["page"]].append(l)
        total_spans = sum(len(v) for v in lines_by_page.values())

        if total_spans <= 80 or len(lines_by_page) <= 2:
            self._compute_statistical_features()  # 🔥 Added here
            self._classify_lines_as_headings()
            heads = [e for e in self.elements if e["cls"] == "HEADING"]
            title = max(heads, key=lambda x: len(x["text"]))["text"] if heads else "Untitled Poster"
            outline = [{"level": "H1", "text": h["text"], "page": h["page"]} for h in heads]
            return json.dumps({"title": title, "outline": outline}, indent=4)

        hf = self._identify_headers_and_footers()
        self.lines = [l for l in self.lines if (l["text"], int(l["bbox"].y0)) not in hf]
        self._compute_statistical_features()
        self._classify_lines_as_headings()
        return json.dumps(self._assemble_final_outline(), indent=4)
