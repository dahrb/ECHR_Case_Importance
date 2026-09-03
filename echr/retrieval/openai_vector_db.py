"""
OpenAI embedding vector DB utilities (reference implementation from original paper).
Migrated from old/VectorDB/openai_vector_db.py.

The API key is read from the OPENAI_API_KEY environment variable.
"""

import os

openai_key = os.environ.get("OPENAI_API_KEY", "")
