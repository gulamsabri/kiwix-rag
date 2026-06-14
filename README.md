# Kiwix RAG

Offline question-answering over a [Kiwix](https://www.kiwix.org/) ZIM library. Ask anything — answers come from your local ZIM collection, served by a quantized LLM via [Ollama](https://ollama.com/). No internet required.

Designed for deployment on a Raspberry Pi 5 with an external SSD. Originally built for off-grid and post-collapse scenarios where the Kiwix library is the only reference available.

## Architecture

```
ZIM files → extract_zim.py → .jsonl chunks → build_index.py → ChromaDB
                                                                    ↓
                                                  query → semantic group routing
                                                                    ↓
                                                         ChromaDB vector search
                                                                    ↓
                                                       Ollama LLM (llama3.2:3b)
                                                                    ↓
                                                           Flask web UI / CLI
```

- **Embedding model**: `all-MiniLM-L6-v2` (sentence-transformers, ~80MB, runs on Pi 5)
- **Vector store**: ChromaDB (persistent, local)
- **LLM**: any Ollama model — `llama3.2:3b` recommended for Pi 5 (~3 t/s)
- **Web UI**: Flask with Server-Sent Events for streaming responses

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running
- One or more Kiwix `.zim` files (download at [library.kiwix.org](https://library.kiwix.org/))

```bash
ollama pull llama3.2:3b
```

## Installation

```bash
python -m venv ~/kiwix-rag
source ~/kiwix-rag/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Extract a ZIM file

```bash
python extract_zim.py path/to/file.zim -o chunks.jsonl
# With OCR for scanned PDFs:
python extract_zim.py path/to/file.zim -o chunks.jsonl --ocr
```

### 2. Build the vector index

```bash
python build_index.py chunks.jsonl --db ./vector_db
```

Repeat for each ZIM. To replace an existing collection, pass `--replace`.

### 3. Query

**CLI (single question):**
```bash
python rag.py "how do I treat a deep wound?" --db ./vector_db
```

**CLI (interactive):**
```bash
python rag.py --db ./vector_db
```

**Web UI:**
```bash
python web.py --db ./vector_db
# Open http://localhost:5000
```

## Web UI options

```
python web.py [options]

  --db PATH              ChromaDB directory (default: ./vector_db next to the script)
  --embed-model PATH     Path to embedding model (default: all-MiniLM-L6-v2 from HF cache)
  --model NAME           Ollama model name (default: phi3:mini)
  --ollama-url URL       Ollama base URL (default: http://localhost:11434)
  --top-k N              Chunks retrieved per query (default: 3)
  --top-groups N         Max collection groups searched per query (default: 2)
  --max-per-group N      Max collections selected per group (default: 15)
  --max-cache-bytes N    Resident byte budget for collection indexes (default: 11000000000 ≈ 11 GB)
  --max-collection-size N  Skip collections larger than this many vectors (default: unlimited)
```

### Memory scaling (Pi 5)

Collection HNSW indexes vary widely in size — from a few MB for small ZIMs up to 6–7 GB for the largest. Rather than capping a fixed collection count, the server bounds total resident index bytes via `--max-cache-bytes` (default ~11 GB); `MemoryMax=13G` in the systemd service is the hard backstop. Example for a 16 GB Pi 5:

```bash
python web.py \
  --db /mnt/ssd/vector_db \
  --embed-model /mnt/ssd/all-MiniLM-L6-v2 \
  --model llama3.2:3b \
  --max-collection-size 500000 \
  --max-cache-bytes 11000000000 \
  --max-per-group 5
```

`--max-cache-bytes 11000000000` caps the resident collection-index byte budget at ~11 GB. With Ollama at ~2.5 GB and OS at ~1 GB, total is ~14.5 GB — safe for a 16 GB Pi 5.

## Semantic group routing

`web.py` routes each query to the most relevant collection groups based on cosine similarity of the query against group descriptions. This dramatically reduces search time and improves answer quality by avoiding irrelevant collections.

Groups are defined in `web.py` at the `GROUPS` dict. Add new patterns there when you index new ZIM libraries.

## Pi deployment

### Systemd services

Copy both `.service` files to the Pi and enable them:

```bash
sudo cp kiwix-rag.service /etc/systemd/system/
sudo cp kiwix-serve.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now kiwix-rag kiwix-serve
```

Edit the `ExecStart` paths in `kiwix-rag.service` to match your venv and SSD mount point.

### Syncing to Pi via SSD

The workflow uses a shared external SSD rather than SSH rsync (faster for large vector DBs):

```bash
# Stop Pi services, eject SSD from Pi, connect to Mac, then:
bash update_pi.sh            # sync vector DB only
bash update_pi.sh --scripts  # also sync web.py, templates, eval.py
bash update_pi.sh --kiwix    # also print the kiwix library rebuild command to run on the Pi
```

### Adding a new ZIM

```bash
# 1. Extract and index on Mac
python extract_zim.py new.zim -o new_chunks.jsonl
python build_index.py new_chunks.jsonl --db ./vector_db

# 2. Add the collection name pattern to the right GROUPS entry in web.py

# 3. Sync to Pi
bash update_pi.sh --scripts
```

## Evaluation

`eval.py` runs a set of test questions against the web UI and grades the answers:

```bash
python eval.py --url http://localhost:5000
# or against the Pi:
python eval.py --url http://meshpi.local:5000
```

## Batch indexing

`batch_index.sh` processes a queue of ZIM files from a manifest, resuming automatically if interrupted:

```bash
cp batch_manifest.example batch_manifest.conf
# edit batch_manifest.conf with your ZIM stems and --ocr flags
bash batch_index.sh batch_manifest.conf 2>&1 | tee batch_live.log
```

Override the ZIM directory or venv path with env vars:

```bash
KIWIX_DIR=/path/to/zims VENV=/path/to/venv/bin/activate bash batch_index.sh batch_manifest.conf
```

## OCR support

For ZIM files containing scanned PDFs:

```bash
# easyocr (pure Python, no binary needed):
pip install easyocr
python extract_zim.py file.zim -o chunks.jsonl --ocr --ocr-engine easyocr

# tesseract (faster):
brew install tesseract   # macOS
pip install pytesseract
python extract_zim.py file.zim -o chunks.jsonl --ocr
```
