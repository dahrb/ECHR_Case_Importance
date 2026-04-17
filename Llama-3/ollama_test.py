import ollama

embedding = ollama.embeddings(model="llama2:7b", prompt="Hello Ollama!")