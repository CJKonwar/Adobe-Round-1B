from documentIntellligence import DocumentIntelligence

di = DocumentIntelligence(
    bi_model_path="models/BAAI/bge-base-en-v1.5",
    reranker_model="models/cross-encoder/ms-marco-MiniLM-L6-v2",
    persona="Travel Planner",
    job_to_be_done="Plan a trip of 4 days for a group of 10 college friends.",
    pdf_dir="/Users/champakjyotikonwar/My_Projects/Adobe-Round-1B/Challenge_1b/Collection 1/PDFs"
    )
di.analyze()

