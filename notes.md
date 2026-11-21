# Notes
```
Type: Questions -> Get Answers
Type: exit -> to Quit
```
## Run
## Terminal 1:
```
ollama serve
```

## Terminal 2:
```
pip install -r requirements.txt -> DON'T DO THIS IF DONE INSTALL
python main.py
```

## Why i chose qwen2:0.5b (352 MB) instead of llama3 (4.7 GB)
### llama3 (4.7 GB) exceeds safe memory on 8 GB MacBook M1 — causes OOM/crash. qwen2:0.5b (352 MB) fits with headroom for macOS, Python, and vector index. All other instruction requirements (LlamaIndex API, HuggingFaceEmbedding BGE-small, Ollama local) preserved.

