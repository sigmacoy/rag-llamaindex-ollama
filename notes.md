# Notes

## Run (2 terminals)

**Terminal 1** — Start Ollama:
```bash
ollama serve
```

**Terminal 2** — Run RAG app:
```bash
cd /Users/mccauleybacalla/SigMacoy/Code/Repos/rag-llamaindex-ollama
python -m pip install -r requirements.txt   # run once (updated deps)
python main.py
```

## Models (for 8GB Mac)
- `qwen2:0.5b` — 352 MB LLM (smallest Ollama model)
- `BAAI/bge-small-en-v1.5` — ~130 MB embeddings (HuggingFace)

## Usage
- Type question → Enter
- Type `exit` → Quit