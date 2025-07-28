from sentence_transformers import SentenceTransformer


# Load and save the bi-encoder model (used for efficient similarity search)
print("Downloading bi-encoder model...")
bi_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
bi_model.save('./models/BAAI/bge-base-en-v1.5')

# Load and save the cross-encoder model (used for accurate reranking of candidates)
print("Downloading cross-encoder model...")
ce_model = SentenceTransformer('cross-encoder/ms-marco-MiniLM-L6-v2')
ce_model.save('./models/cross-encoder/ms-marco-MiniLM-L6-v2')

print("All models downloaded successfully!")
