import os
import json
from documentIntellligence import DocumentIntelligence

INPUT_JSON = "/app/input.json"  
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"

def get_output_filename(input_json):
    if input_json.endswith('_input.json'):
        return input_json.replace('_input.json', '_output.json')
    elif input_json.endswith('.json'):
        base = input_json.rsplit('.', 1)[0]
        return base + '_output.json'
    else:
        return "analysis_output.json"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_JSON, "r") as f:
        config = json.load(f)

    OUTPUT_FILENAME = get_output_filename(os.path.basename(INPUT_JSON))
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    persona = config["persona"]["role"]
    job = config["job_to_be_done"]["task"]

    pdf_filenames = [doc["filename"] for doc in config["documents"]]
    pdf_files = [
        os.path.join(INPUT_DIR, f)
        for f in pdf_filenames
        if os.path.isfile(os.path.join(INPUT_DIR, f))
    ]
    if not pdf_files:
        print(f"No matching PDF files found in {INPUT_DIR}. Files expected: {pdf_filenames}")
        return

    di = DocumentIntelligence(
        bi_model_path="/app/models/BAAI/bge-base-en-v1.5",
        reranker_model="/app/models/cross-encoder/ms-marco-MiniLM-L6-v2",
        persona=persona,
        job_to_be_done=job
    )
    di.analyze(
        pdf_files=pdf_files,
        output_path=OUTPUT_PATH
    )
    print(f"✅ All results written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
