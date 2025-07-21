import os
import json
import datetime
import numpy as np
import fitz
from sentence_transformers import SentenceTransformer, CrossEncoder
from extraction_1A import SmartPDFOutline

# Hyperparameters
CHUNK_TOKENS      = 256
CHUNK_STRIDE      = 64
STAGE_A_POOL      = 50
TOP_K_SECTIONS    = 5
TOP_K_SUBSECTIONS = 5
MMR_LAMBDA        = 0.6


class DocumentIntelligence:
    def __init__(self,
                 bi_model_path: str,
                 reranker_model: str,
                 persona: str,
                 job_to_be_done: str,
                 pdf_dir: str):
     
        self.persona = persona
        self.job = job_to_be_done
        self.pdf_dir = pdf_dir
        self.documents = [
            f for f in os.listdir(pdf_dir)
            if f.lower().endswith('.pdf')
        ]
        # Stage A: bi-encoder for fast candidate generation
        self.bi = SentenceTransformer(bi_model_path, device="cpu")
        # Stage B: cross-encoder for precise reranking
        self.ce = CrossEncoder(reranker_model, device="cpu", max_length=512)

    def analyze(self,top_k_sections: int = TOP_K_SECTIONS,top_k_subsections: int = TOP_K_SUBSECTIONS):
        # 1. Build query embedding
        query = f"{self.persona}. {self.job}"
        query_emb = self.bi.encode(query, normalize_embeddings=True)

        # 2. Extract headings & chunk text
        chunks = self._extract_and_chunk_all()
        if not chunks:
            return self._empty_output("No chunks extracted")

        # 3. Stage A – bi-encoder scoring
        emb_matrix = np.vstack([c["emb"] for c in chunks])
        scores_a = emb_matrix @ query_emb
        cand_idx = np.argsort(scores_a)[::-1][:STAGE_A_POOL]

        # 4. Stage B – cross-encoder reranking
        pairs = [(query, chunks[i]["text"]) for i in cand_idx]
        scores_b = self.ce.predict(pairs, batch_size=32, convert_to_numpy=True)

        # 5. MMR diversity selection for top sections
        selected = self._mmr_select(
            query_emb,
            [chunks[i]["emb"] for i in cand_idx],
            scores_b,
            mmr_lambda=MMR_LAMBDA,
            top_k=top_k_sections
        )
        final_section_indices = [cand_idx[i] for i in selected]

        # 6. Build extracted_sections payload
        extracted_sections = []
        for rank, idx in enumerate(final_section_indices, start=1):
            c = chunks[idx]
            extracted_sections.append({
                "document": c["doc"],
                "section_title": c["heading"],
                "importance_rank": rank,
                "page_number": c["page"]
            })

        # 7. Building subsection_analysis payload
        all_pairs = [(query, c["text"]) for c in chunks]
        all_scores = self.ce.predict(all_pairs, batch_size=64, convert_to_numpy=True)
        top_sub = np.argsort(all_scores)[::-1][:top_k_subsections]
        subsection_analysis = []
        for idx in top_sub:
            c = chunks[idx]
            subsection_analysis.append({
                "document": c["doc"],
                "refined_text": c["text"][:500],
                "page_number": c["page"]
            })

        # 8. Final output
        output = {
            "metadata": {
                "input_documents": self.documents,
                "persona": self.persona,
                "job_to_be_done": self.job,
                "processing_timestamp": datetime.datetime.now().isoformat()
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
        with open("analysis_output.json", "w") as f:
            json.dump(output, f, indent=2)
        print("Analysis complete. Results saved to analysis_output.json")
        return output

    def _extract_and_chunk_all(self):
        chunks = []
        for doc_name in self.documents:
            pdf_path = os.path.join(self.pdf_dir, doc_name)
            extractor = SmartPDFOutline(pdf_path)
            outline = json.loads(extractor.analyze()).get("outline", [])
            doc = fitz.open(pdf_path)
            headings_by_page = {}
            for h in outline:
                headings_by_page.setdefault(h["page"], []).append(h["text"])

            for page_idx in range(len(doc)):
                text = doc[page_idx].get_text("text")
                windows = self._token_windows(text, CHUNK_TOKENS, CHUNK_STRIDE)
                for win in windows:
                    heading = (headings_by_page.get(page_idx) or [f"Page-{page_idx+1}"])[0]
                    chunks.append({
                        "doc": doc_name,
                        "page": page_idx + 1,
                        "heading": heading,
                        "text": win,
                        "emb": None
                    })
            doc.close()
        # Batch-encode text embeddings
        texts = [c["text"] for c in chunks]
        embs = self.bi.encode(texts, batch_size=64, normalize_embeddings=True)
        for c, e in zip(chunks, embs):
            c["emb"] = e
        return chunks
    #Simple whitespace-based token windows.
    @staticmethod
    def _token_windows(text: str, window: int, stride: int):
        words = text.split()
        if len(words) <= window:
            return [" ".join(words)]
        out = []
        step = window - stride
        for i in range(0, len(words), step):
            chunk = words[i:i+window]
            out.append(" ".join(chunk))
            if i + window >= len(words):
                break
        return out

    # Apply Maximum Marginal Relevance to cand_scores+embs.
    def _mmr_select(self, query_emb, cand_embs, cand_scores, mmr_lambda, top_k):
        cand_embs = np.vstack(cand_embs)
        sim = cand_embs @ cand_embs.T
        candidate_ids = list(range(len(cand_scores)))
        selected = []
        while candidate_ids and len(selected) < top_k:
            if not selected:
                best = candidate_ids[int(np.argmax(cand_scores[candidate_ids]))]
            else:
                mmr_vals = []
                for i in candidate_ids:
                    rel = cand_scores[i]
                    div = max(sim[i, selected])
                    mmr_vals.append(mmr_lambda * rel - (1 - mmr_lambda) * div)
                best = candidate_ids[int(np.argmax(mmr_vals))]
            selected.append(best)
            candidate_ids.remove(best)
        return selected

    def _empty_output(self, err_msg: str):
        return {
            "metadata": {
                "input_documents": self.documents,
                "persona": self.persona,
                "job_to_be_done": self.job,
                "processing_timestamp": datetime.datetime.now().isoformat(),
                "error": err_msg
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }
