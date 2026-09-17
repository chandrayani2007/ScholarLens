import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import logging
from dotenv import load_dotenv
load_dotenv()

from src.pipeline.llm import get_llm_provider
from src.pipeline.rag import RAGPipeline

logging.basicConfig(level=logging.INFO)

print('LLM_PROVIDER:', os.environ.get('LLM_PROVIDER'))
print('GEMINI_API_KEY exists:', bool(os.environ.get('GEMINI_API_KEY')))

llm = get_llm_provider('gemini')
print('Testing gemini directly...')
try:
    ans = llm.generate('Hello, return JSON: {"status": "ok"}')
    print('Gemini response:', ans[:100])
except Exception as e:
    print('Gemini failed:', e)

print('Testing RAGPipeline with Gemini...')
pipe = RAGPipeline(llm_provider=llm)
res = pipe.answer('What research problem does this paper address?', filters={'paper_id': 'AI038'})
print('Result answer:\n', res.answer)
print('Result citations:\n', res.citations)
print('Result evidence count:\n', len(res.evidence))
