from sentence_transformers import SentenceTransformer
model = SentenceTransformer('cross-encoder/ms-marco-MiniLM-L6-v2')
model.save('cross-encoder/ms-marco-MiniLM-L6-v2')
