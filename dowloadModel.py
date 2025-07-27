from sentence_transformers import SentenceTransformer
model = SentenceTransformer('cross-encoder/ms-marco-MiniLM-L6-v2')
model = SentenceTransformer('BAAI/bge-base-en-v1.5')
model.save('cross-encoder/ms-marco-MiniLM-L6-v2')
model.save('BAAI/bge-base-en-v1.5')
