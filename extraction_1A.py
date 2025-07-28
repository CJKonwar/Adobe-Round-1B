import os
import fitz  
import json
import numpy as np
import re
import bisect
import itertools
from collections import defaultdict
from thefuzz import fuzz

class SmartPDFOutline:
    def __init__(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.lines = []
        self.features = {}
        self.elements = []
        self.semantic_heading_patterns = re.compile(
            r'^(abstract|introduction|summary|background|overview|conclusion|references|appendix|contents|figure|table|chapter|section|part|phase|postscript|preamble|glossary|index|acknowledgements|methodology|results|discussion|evaluation|approach|requirements|milestones|timeline|membership|terms of reference)\b|'
            r'^\d+(\.\d+)*\s+[A-Z]',
            re.IGNORECASE
        )

    @staticmethod
    def is_poster_doc(lines_by_page, max_spans=80, max_pages=2):
        total_spans = sum(len(v) for v in lines_by_page.values())
        return total_spans <= max_spans or len(lines_by_page) <= max_pages

    @staticmethod
    def classify_poster(lines_by_page):
        # For each y-position (row), concatenate horizontally, treat high font size as heading
        all_lines = [line for lines in lines_by_page.values() for line in lines]
        if not all_lines:
            return [], ""
        sizes = [line["size"] for line in all_lines if line["text"]]
        if not sizes:
            return [], ""
        m, sd, md = np.mean(sizes), np.std(sizes), np.median(sizes)
        elements = []
        # Sort by page then y
        all_lines.sort(key=lambda x: (x["page"], round(x["bbox"].y1)))
        from itertools import groupby
        for (pg, y), group in groupby(all_lines, key=lambda x: (x["page"], round(x["bbox"].y1))):
            row = list(group)
            row = sorted(row, key=lambda x: x["bbox"].x0)
            txt = " ".join(r["text"] for r in row).strip()
            if not txt or re.fullmatch(r"\W+", txt):
                continue
            size = max(r["size"] for r in row)
            bold = any(r.get("font", "").lower().find("bold") != -1 for r in row)
            is_heading = size > m + 1.5*sd or (size > md * 1.2 and bold)
            elements.append({
                "text": txt,
                "page": pg,
                "level": "H1" if is_heading else "BODY_TEXT"
            })
        # Title = longest text among headings
        headings = [el for el in elements if el["level"] == "H1"]
        title = max(headings, key=lambda x: len(x["text"]))["text"] if headings else (elements[0]["text"] if elements else "")
        return [el for el in elements if el["level"].startswith("H")], title

    def _is_semantic_heading(self, text):
        stripped = text.strip()
        return bool(self.semantic_heading_patterns.match(stripped) and len(stripped.split()) < 12)

    def _build_lines_with_features(self):
        for pnum, page in enumerate(self.doc):
            page_width = page.rect.width
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_LIGATURES)["blocks"]
            for block in blocks:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    spans = [s for s in line["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    line_bbox = fitz.Rect(line["bbox"])
                    x_center = (line_bbox.x0 + line_bbox.x1) / 2
                    center_offset = abs((page_width / 2) - x_center)
                    centeredness = max(0, 1 - (center_offset / (page_width / 2)))
                    avg_size = sum(s["size"] for s in spans) / len(spans)
                    is_bold = any(s["flags"] & 16 for s in spans)
                    full_text = " ".join(s["text"] for s in spans).strip()
                    is_semantic = self._is_semantic_heading(full_text)
                    self.lines.append({
                        "text": full_text,
                        "size": avg_size,
                        "bold": is_bold,
                        "page": pnum,
                        "bbox": line_bbox,
                        "centeredness": centeredness,
                        "is_semantic": is_semantic
                    })
    
    def _identify_headers_and_footers(self, top_n=5, bottom_n=3, threshold=75, tol=10):
        pages = defaultdict(list)
        for line in self.lines:
            pages[line["page"]].append(line)

        sorted_pages = {}
        for pnum, lines in pages.items():
            lines.sort(key=lambda l: l["bbox"].y0)
            sorted_pages[pnum] = {
                "lines": lines,
                "y_coords": [l["bbox"].y0 for l in lines]
            }

        candidates = set()
        for pnum in range(1, len(self.doc)):
            prev = sorted_pages.get(pnum - 1)
            curr = sorted_pages.get(pnum)
            if not prev or not curr:
                continue
            def match_lines(curr_lines, prev_data):
                for ln in curr_lines:
                    y = ln["bbox"].y0
                    lo = bisect.bisect_left(prev_data["y_coords"], y - tol)
                    hi = bisect.bisect_right(prev_data["y_coords"], y + tol)
                    for i in range(lo, hi):
                        pl = prev_data["lines"][i]
                        if fuzz.ratio(ln["text"], pl["text"]) > threshold:
                            candidates.add((ln["text"], int(y)))
                            candidates.add((pl["text"], int(pl["bbox"].y0)))
                            break
            match_lines(curr["lines"][:top_n], prev)
            match_lines(curr["lines"][-bottom_n:], prev)
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
            is_heading = (
                (l["is_semantic"] and (l["bold"] or l["size"] > md * 1.05)) or
                (l["bold"] and l["size"] > md * 1.15) or
                (l["size"] > m + 1.8 * sd)
            )
            if is_heading and len(l["text"].split()) <= 20:
                self.elements.append({**l, "cls": "HEADING"})

    def _bucket_headings_by_level(self, el):
        size, bold, centered = el["size"], el["bold"], el["centeredness"]
        m, sd, md = (self.features[k] for k in ("mean_size", "std_dev_size", "median_size"))
        if size > m + 1.5 * sd and bold: return "H1"
        if size > m + 1.0 * sd and centered > 0.7: return "H1"
        if bold and size > m: return "H2"
        if bold or (el["is_semantic"] and size > md): return "H3"
        return "H4"

    def _assemble_final_outline(self):
        page0 = [l for l in self.lines if l["page"] == 0]
        candidates = sorted(page0, key=lambda l: (l["size"], len(l["text"])), reverse=True)
        title = next((c["text"].strip() for c in candidates if len(c["text"]) > 5), "Untitled Document")
        title_set = {line["text"] for line in candidates[:2]}
        heads = [e for e in self.elements if e["text"] not in title_set]

        outline = []
        for h in heads:
            level = self._bucket_headings_by_level(h)
            outline.append({"level": level, "text": h["text"], "page": h["page"]})

        order = {"H1": 1, "H2": 2, "H3": 3, "H4": 4}
        outline.sort(key=lambda x: (x["page"], order.get(x["level"], 99)))
        final = [next(g) for _, g in itertools.groupby(outline, key=lambda x: (x["text"], x["page"]))]
        return {"title": title, "outline": final}

    def analyze(self):
        self._build_lines_with_features()
        # --- Poster mode check ---
        # Group lines by page for poster detection
        lines_by_page = defaultdict(list)
        for l in self.lines:
            lines_by_page[l["page"]].append(l)
        if self.is_poster_doc(lines_by_page):
            poster_headings, title = self.classify_poster(lines_by_page)
            outline = []
            for el in poster_headings:
                # Try to assign level based on size (since poster mode is flat, everything H1)
                outline.append({"level": el.get("level", "H1"), "text": el["text"], "page": el["page"]})
            return json.dumps({"title": title, "outline": outline}, indent=4)
        # --- Standard path ---
        headerfooters = self._identify_headers_and_footers()
        self.lines = [l for l in self.lines if (l["text"], int(l["bbox"].y0)) not in headerfooters]
        self._compute_statistical_features()
        self._classify_lines_as_headings()
        return json.dumps(self._assemble_final_outline(), indent=4)
