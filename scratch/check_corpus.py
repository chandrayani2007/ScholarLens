import os
from src.pipeline.retrieval import HybridRetriever

retriever = HybridRetriever()
print("Chroma collection count:", retriever.collection.count() if retriever.collection else "None")

# Search for Deepchecks or RAG
res = retriever.retrieve("Deepchecks Evaluating Retrieval Augmented Generation", top_k=5)
for r in res:
    print(f"Paper ID: {r.paper_id} | Section: {r.section_name} | Text: {r.text[:100]}...")
