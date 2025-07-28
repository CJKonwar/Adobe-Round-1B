# **Persona-Driven Document Intelligence Solution (Round 1B)**

---

### **Challenge Overview**
This solution addresses the **Persona-Driven Document Intelligence** challenge, where the goal is to build an intelligent document analyst that extracts and prioritizes the most relevant sections from a collection of documents based on a specific **persona** and their **job-to-be-done**. The system must generalize across diverse domains (**academic research**, **business analysis**, **educational content**) while operating under strict computational constraints.

### **Problem Statement**
Traditional document analysis treats all content equally, but real-world users have specific roles, expertise levels, and objectives. A **PhD researcher** analyzing **drug discovery** papers needs different information than an **investment analyst** reviewing **financial reports**. Our solution bridges this gap by implementing **contextual relevance ranking** tailored to user personas and their specific tasks.

### **Approach Overview**
This solution implements a sophisticated **two-stage ranking system** combined with **diversity optimization** to extract the most contextually relevant document sections. The architecture balances **computational efficiency** with **precision**, ensuring optimal performance within the **1GB memory constraint** and **60-second processing limit**.

---

## **Core Methodology**

### **1. Foundation: Document Processing & Intelligent Chunking**

#### **Structure-Aware Extraction**
- Leverages the **Round 1A solution (SmartPDFOutline)** to extract document hierarchies, titles, and heading structures
- Preserves document organization and **semantic relationships** between sections
- Maintains **page-level mapping** for precise source attribution

#### **Advanced Sliding Window Chunking**
- Implements **token-based text segmentation** with carefully tuned parameters:
  - **Window Size**: 256 tokens (optimal balance between context and processing speed)
  - **Stride**: 64 tokens (25% overlap ensures no information loss at boundaries)
- Maintains **contextual coherence** while ensuring comprehensive document coverage
- Each chunk inherits **metadata** from its parent section for enhanced relevance scoring

#### **Semantic Context Preservation**
- Maps each text chunk to its corresponding document **section and heading**
- Preserves **hierarchical relationships** (**H1 → H2 → H3**) for better context understanding
- Enables **section-aware relevance scoring and ranking**

---

### **2. Two-Stage Retrieval Architecture**
Our hybrid retrieval system combines the **speed of bi-encoders** with the **precision of cross-encoders**, inspired by modern information retrieval research.

#### **Stage A: Bi-Encoder Candidate Generation**
- **Model**: SentenceTransformer for efficient semantic encoding (BAAI/bge-base-en-v1.5)
- **Query Construction**: Combines persona description with job-to-be-done for comprehensive context
- **Similarity Computation**: Cosine similarity between query embedding and document chunk embeddings
- **Candidate Pool**: Selects top **50 candidates** (configurable via `STAGE_A_POOL`)
- **Optimization**: Batch processing with **normalized embeddings** for consistent scoring

#### **Stage B: Cross-Encoder Reranking**
- **Model**: CrossEncoder for nuanced query-document pair scoring (cross-encoder/ms-marco-MiniLM-L6-v2)
- **Input Processing**: Query-chunk pairs processed with **512 token limit** for optimal performance
- **Batch Processing**: Configurable batch size (**32 for candidate reranking**, **64 for subsection analysis**)
- **Precision Focus**: Captures **subtle semantic relationships** missed by bi-encoder similarity()

---

### **3. Maximum Marginal Relevance (MMR) Diversity Selection**

#### **Algorithm Implementation**
- **Lambda Parameter**: 0.6 (**60% relevance weight**, **40% diversity weight**)
- **Diversity Mechanism**: Prevents selection of redundant sections by measuring **inter-document similarity**
- **Selection Strategy**: Iteratively selects documents **balancing relevance score with dissimilarity** to already selected content

#### **Benefits**
- Ensures **comprehensive coverage** across different document aspects
- Prevents **information redundancy** in final output
- Maintains **high relevance** while maximizing **information diversity**

---

### **4. Dual Output Generation Strategy**

#### **Primary Output: Extracted Sections**
- **Selection Criteria**: Top-K sections (default: 5) based on **MMR scores**
- **Ranking System**: **Importance-based ranking** with clear hierarchical ordering
- **Metadata Preservation**: Document name, section title, page number, and importance rank
- **Context Maintenance**: Links back to **original document structure**

#### **Secondary Output: Subsection Analysis**
- **Granular Extraction**: Fine-grained text segments for detailed analysis
- **Length Filtering**: Minimum **50 characters** to ensure meaningful content
- **Text Refinement**: Truncated to **500 characters** for focused insights
- **Comprehensive Scoring**: All chunks scored for thorough analysis
