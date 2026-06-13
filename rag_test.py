# from app.services.embeddings import embed_texts
# from app.services import vectorstore as V
# import time

# DOC_ID = 99999
# chunks = [
#     "The Eiffel Tower is 330 metres tall and is located in Paris, France.",
#     "The Great Wall of China stretches over 21,000 kilometres.",
#     "The Amazon rainforest produces roughly 20 percent of the world's oxygen.",
# ]
# vecs = embed_texts(chunks)
# V.ensure_index()
# base_md = {"title": "World Facts", "doc_type": "txt", "owner_id": 2, "tags": [], "upload_ts": 0}
# n = V.upsert_chunks(DOC_ID, chunks, vecs, base_md)
# print(f"Indexed {n} chunks as document {DOC_ID}.")
# time.sleep(4)  # serverless eventual consistency
# print("Now POST /api/search (as admin) with: {\"question\": \"How tall is the Eiffel Tower?\"}")



# cleanup rag_test
from app.services import vectorstore as V
V.delete_document(99999)
print("Removed test document 99999.")