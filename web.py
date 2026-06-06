#!/usr/bin/env python3
"""
Web interface for the Kiwix RAG system.
Serves a chat UI and streams answers via Server-Sent Events.
Uses semantic routing to search only the most relevant collection groups
per query, improving both answer quality and response speed.

Usage:
    python web.py
    python web.py --db /mnt/ssd/vector_db --embed-model /mnt/ssd/all-MiniLM-L6-v2
"""

import argparse
import json
import os
import sqlite3
import sys
import time
import threading
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import chromadb
import requests
from flask import Flask, Response, render_template, request
from sentence_transformers import SentenceTransformer

# ── defaults ──────────────────────────────────────────────────────────────────

DB_PATH        = Path(__file__).parent / "vector_db"
EMBED_MODEL    = "all-MiniLM-L6-v2"
OLLAMA_URL     = "http://localhost:11434"
LLM_MODEL      = "phi3:mini"
TOP_K          = 3   # chunks retrieved per query
TOP_GROUPS     = 2   # max groups to search per query
ROUTE_THRESH   = 0.20  # min cosine similarity to pick a group (else search all)
GROUP_TTL      = 600   # seconds before an idle collection is evicted from memory cache

# ── memory scaling TODOs ──────────────────────────────────────────────────────
# TODO option 2: quantize embeddings float32 → int8 at index time (build_index.py)
#   4× memory reduction per collection; requires full re-index of all collections.
# TODO option 3: migrate vector store from ChromaDB to Qdrant with mmap storage
#   Collections live on SSD, only hot pages in RAM; no re-index needed, schema migration required.
#   Once done, --max-collection-size limit can be relaxed or removed.
# TODO option 4: add NVMe SSD to Pi 5 via PCIe HAT (~$15-30)
#   Pairs with option 3; makes mmap storage fast enough for production use.

SYSTEM_PROMPT = (
    "You are a reference assistant for survivors in a post-collapse world where "
    "civilization's infrastructure — hospitals, governments, supply chains, the internet, "
    "emergency services — no longer exists or cannot be reached. "
    "The people using this system are trying to survive, keep others alive, and slowly "
    "rebuild functional communities from whatever remains. Every question they ask is "
    "asked in that spirit: learning to grow food, treat injuries, repair tools, "
    "understand the natural world, preserve knowledge, and pass skills on to others. "
    "Rebuilding requires recovering ALL of human knowledge — including aviation, medicine, "
    "law, engineering, science, mathematics, computing, history, and the arts. "
    "A question about aviation rules, programming, chemistry, or any technical subject "
    "is just as valid as a question about farming or first aid. Survivors may have "
    "access to aircraft, radios, generators, and other technology. They need to "
    "understand how these things work in order to use, repair, and teach others. "
    "Answer every question as if the person asking has no access to any outside help "
    "and must act on the information themselves, right now or in the future. "
    "Answer using only the context passages provided. "
    "Give complete, direct, actionable answers. Never tell users to see a doctor, "
    "call for help, or consult a professional — none of those options exist. "
    "Never refuse to answer. These people are on their own and this knowledge may "
    "be the difference between life and death, sickness and health, despair and hope. "
    "When your answer goes beyond what the provided context explicitly states — drawing "
    "on general principles, filling gaps by inference, or combining information across "
    "sources — say so clearly: for example, 'The sources don't directly address this, "
    "but based on the information provided...' or 'This is my interpretation of the "
    "context, not an explicit instruction from the source.' Never invent specific facts, "
    "figures, doses, or procedures that are not present in the context."
)

# ── collection groups ─────────────────────────────────────────────────────────
# Each group has:
#   description  — rich natural-language text used for semantic routing
#   patterns     — substrings matched against collection names (case-sensitive)
#                  Add new patterns here when future ZIM libraries are indexed.
#
# A collection is assigned to every group whose patterns it matches, so
# collections that span topics (e.g. military medicine) appear in multiple
# groups. Duplicate results are deduplicated at retrieval time.
#
# IMPORTANT: patterns match against collection names, which use underscores
# throughout (dots and hyphens in ZIM/JSONL filenames are converted to _).
# Use "wwwnc_cdc_gov" not "wwwnc.cdc.gov", "health_stackexchange" not
# "health.stackexchange", etc.

GROUPS = {
    "medicine": {
        "description": (
            "How do I treat this wound or injury? What is the correct dose of "
            "this medication? What are the symptoms of this condition? How do I "
            "perform first aid? What does this drug interact with? How do I "
            "diagnose this illness? How do I care for a patient? What causes "
            "this disease? How do I stop bleeding? Is this medication safe?"
        ),
        "patterns": [
            "health_stackexchange",
            "medlineplus",
            "nhs_uk",
            "fas_military_medicine",
            "zimgit_medicine",
            "quickguidesformedicine",
            "wwwnc_cdc_gov",
            "biology_stackexchange",
            "libretexts_org_en_med",
            # future: merck, who_guidelines, tabers
        ],
    },
    "survival": {
        "description": (
            "How do I survive in the wilderness? How do I purify water? How do "
            "I tie this knot? How do I start a fire without matches? How do I "
            "build a shelter? How do I find food in the wild? How do I signal "
            "for rescue? How do I navigate without a compass? What should I do "
            "in a disaster? How do I prepare an emergency kit? How do I stay "
            "warm in winter? How do I stockpile food for emergencies?"
        ),
        "patterns": [
            "zimgit_knots",
            "zimgit_water",
            "zimgit_post_disaster",
            "zimgit_medicine",
            "outdoors_stackexchange",
            "canadian_prepper",
            "urban_prepper",
            "sustainability_stackexchange",
            "martialarts_stackexchange",
            "solar_lowtechmagazine",
            "lifehacks_stackexchange",
            # future: survivorlibrary, army_survival_fm, sere
        ],
    },
    "military": {
        "description": (
            "What does this army field manual say? What is the military "
            "procedure for this? How do soldiers train for this? What is the "
            "doctrine for this operation? What are the military regulations on "
            "this? How do I perform this soldier skill? What is the chain of "
            "command for this? What are the rules of engagement?"
        ),
        "patterns": [
            "fas_military_medicine",
            "armypubs",
            # future: armypubs_en_all, mil_doctrine, field_manuals
        ],
    },
    "coding": {
        "description": (
            "How do I write this function? What does this error mean? How do I "
            "use this library? How do I implement this algorithm? How do I "
            "debug my code? How do I use git? How do I undo a commit? How do I "
            "work with files? What is the syntax for this? How do I compile "
            "this program? How do I write a script to automate this?"
        ),
        "patterns": [
            "devdocs_en_python",
            "devdocs_en_c_",
            "devdocs_en_cpp",
            "devdocs_en_rust",
            "devdocs_en_go_",
            "devdocs_en_erlang",
            "devdocs_en_php",
            "devdocs_en_bash",
            "devdocs_en_cmake",
            "devdocs_en_gcc",
            "devdocs_en_git",
            "devdocs_en_qt",
            "devdocs_en_pygame",
            "devdocs_en_gnuplot",
            "devdocs_en_scikit_image",
            "raspberrypi_stackexchange",
            "softwareengineering_stackexchange",
            "docs_python_org",
            "askubuntu",
            "android_stackexchange",
            "apple_stackexchange",
            "cs_stackexchange",
            "cstheory_stackexchange",
            "emacs_stackexchange",
            "vi_stackexchange",
            "retrocomputing_stackexchange",
            "reverseengineering_stackexchange",
            "engineering_stackexchange",
            # future: stackoverflow, github_docs, language_specs
        ],
    },
    "web": {
        "description": (
            "How do I style this with CSS? How do I use this React hook? How "
            "do I make an API request in JavaScript? How do I build a "
            "responsive layout? How do I handle a form submission? How do I "
            "use TypeScript types? How do I manage state in React? How do I "
            "animate this element? How do I bundle my web app?"
        ),
        "patterns": [
            "devdocs_en_html",
            "devdocs_en_css",
            "devdocs_en_javascript",
            "devdocs_en_typescript",
            "devdocs_en_react_",
            "devdocs_en_react_bootstrap",
            "devdocs_en_react_native",
            "devdocs_en_react_router",
            "devdocs_en_node",
            "devdocs_en_webpack",
            "devdocs_en_rxjs",
            "devdocs_en_axios",
            "devdocs_en_redux",
            "devdocs_en_socketio",
            "devdocs_en_date_fns",
            "devdocs_en_moment",
            "devdocs_en_less",
            "craftcms_stackexchange",
            # future: mdn_web_docs, web_components
        ],
    },
    "devops": {
        "description": (
            "How do I configure nginx or Apache? How do I deploy a Docker "
            "container? How do I set up Kubernetes? How do I manage a "
            "database? How do I configure a server? How do I set up a reverse "
            "proxy? How do I manage DNS or networking? How do I automate "
            "deployment? How do I monitor my services? How do I set up a VPN?"
        ),
        "patterns": [
            "devdocs_en_docker",
            "devdocs_en_kubernetes",
            "devdocs_en_nginx",
            "devdocs_en_redis",
            "devdocs_en_apache",
            "devdocs_en_terraform",
            "devdocs_en_kubectl",
            "devdocs_en_nix",
            "devdocs_en_postgresql",
            "devdocs_en_mariadb",
            "devdocs_en_npm",
            "networkengineering_stackexchange",
            "serverfault",
            "askubuntu",
            "dba_stackexchange",
            # future: ansible_docs, prometheus, grafana
        ],
    },
    "data": {
        "description": (
            "How do I train a machine learning model? How do I process a "
            "dataset with pandas? How do I plot this data? How do I build a "
            "neural network? How do I evaluate model accuracy? How do I do "
            "statistical analysis? How do I use numpy arrays? How do I "
            "preprocess data for ML? How do I use PyTorch or TensorFlow?"
        ),
        "patterns": [
            "devdocs_en_numpy",
            "devdocs_en_pandas",
            "devdocs_en_matplotlib",
            "devdocs_en_scikit_learn",
            "devdocs_en_tensorflow",
            "devdocs_en_pytorch",
            # future: huggingface_docs, kaggle_guides, scipy
        ],
    },
    "physics": {
        "description": (
            "What is the speed of light? How does gravity work? What is quantum "
            "mechanics? How do electromagnetic waves propagate? What is relativity? "
            "How does electricity flow? What is thermodynamics? What is energy? "
            "How do magnets work? What is a force? How does nuclear fission work? "
            "What is momentum? How does optics work? What is a wave?"
        ),
        "patterns": [
            "physics_stackexchange",
            "libretexts_org_en_phys",
        ],
    },
    "chemistry": {
        "description": (
            "How does this chemical reaction work? What is the molecular structure "
            "of this compound? How do I balance this equation? What is an acid or "
            "base? How do I identify this element? What is oxidation? How do bonds "
            "form? What is the periodic table? How do I make this compound? "
            "What is thermochemistry? How does electrochemistry work?"
        ),
        "patterns": [
            "chemistry_stackexchange",
            "libretexts_org_en_chem",
        ],
    },
    "biology": {
        "description": (
            "How does DNA work? What is natural selection? How does photosynthesis "
            "work? What is the cell cycle? How do viruses and bacteria replicate? "
            "What is an ecosystem? How does the immune system work? What is "
            "genetics? How do organisms evolve? What is metabolism? How does the "
            "nervous system work? How do plants grow? What is taxonomy?"
        ),
        "patterns": [
            "biology_stackexchange",
            "libretexts_org_en_bio",
        ],
    },
    "mathematics": {
        "description": (
            "What is the central limit theorem? How do I solve this integral? "
            "What is a probability distribution? How do I calculate a confidence "
            "interval? What is a hypothesis test? How does linear algebra work? "
            "What is a differential equation? How do I prove this theorem? "
            "What is a normal distribution? How do I do statistical analysis? "
            "What is a derivative? How does numerical computation work? "
            "What is set theory? How do I solve this equation?"
        ),
        "patterns": [
            "stacks_math_columbia_edu",
            "libretexts_org_en_math",
            "libretexts_org_en_stats",
            "scicomp_stackexchange",
        ],
    },
    "earth_science": {
        "description": (
            "How do tectonic plates work? How do volcanoes form? What causes "
            "earthquakes? How does weather work? What is the water cycle? "
            "How do stars form? What is a black hole? What are the planets? "
            "How does the atmosphere work? What causes climate? How do glaciers "
            "form? What is geology? How do I identify this rock or mineral? "
            "What causes ocean currents? How do seasons work?"
        ),
        "patterns": [
            "earthscience_stackexchange",
            "astronomy_stackexchange",
            "space_stackexchange",
            "libretexts_org_en_geo",
        ],
    },
    "gardening": {
        "description": (
            "How do I grow this plant? Why are my plants dying? How do I "
            "control pests in my garden? When should I plant this vegetable? "
            "How do I improve my soil? What is companion planting? How do I "
            "compost? How do I prune this plant? How do I save seeds? What "
            "is wrong with my tomatoes? How do I grow food in small spaces?"
        ),
        "patterns": [
            "gardening",
            "sustainability_stackexchange",
            # future: permaculture, square_foot_gardening, rhs
        ],
    },
    "cooking": {
        "description": (
            "How do I cook this dish? What is the recipe for this? How long do "
            "I bake this? How do I substitute this ingredient? Why did my dish "
            "turn out wrong? What temperature should I use? How do I know when "
            "this is done cooking? How do I store leftovers safely? What is "
            "this cooking technique called? How do I brew beer? What are the "
            "nutritional guidelines for this food?"
        ),
        "patterns": [
            "cooking_stackexchange",
            "based_cooking",
            "grimgrains",
            "usda_2015",
            "beer_stackexchange",
            "alcohol_stackexchange",
            # future: recipe_databases, fdc_usda
        ],
    },
    "automotive": {
        "description": (
            "Why is my car making this noise? How do I change the oil? What is "
            "wrong with my engine? How do I replace this part? Why won't my "
            "car start? What does this warning light mean? What are the "
            "symptoms of this car problem? How do I fix this brake issue? "
            "How do I diagnose this vehicle fault? How do I do this repair?"
        ),
        "patterns": [
            "mechanics_stackexchange",
            # future: haynes_manuals, alldata, vehicle_repair
        ],
    },
    "aviation": {
        "description": (
            "What is VFR and IFR flying? How do I read aviation charts? "
            "What are the rules for flying under visual flight rules? "
            "How does instrument flight work? What is a crosswind landing? "
            "How do aircraft engines work? What is a transponder? "
            "What are the requirements for a pilot's license? "
            "How do I file a flight plan? What is ATC communication? "
            "How does an altimeter work? What causes an aerodynamic stall?"
        ),
        "patterns": [
            "aviation_stackexchange",
        ],
    },
    "repair": {
        "description": (
            "How do I fix this broken device? How do I replace this screen or "
            "battery? How do I open this appliance to repair it? What tools do "
            "I need for this repair? How do I solder this component? How do I "
            "fix this home appliance? How do I do this DIY home improvement? "
            "What are the steps to repair this? How do I 3D print a replacement part?"
        ),
        "patterns": [
            "ifixit",
            "diy_stackexchange",
            "crafts_stackexchange",
            "3dprinting_stackexchange",
            # future: repair_cafe, fixya
        ],
    },
    "security": {
        "description": (
            "How do I secure this system? How does this attack work? How do I "
            "test for this vulnerability? How do I set up a firewall? How do I "
            "implement authentication securely? What is this CVE? How do I "
            "detect this malware? How do I do penetration testing? How do I "
            "encrypt this data? How do I harden this server?"
        ),
        "patterns": [
            "security_stackexchange",
            "reverseengineering_stackexchange",
            "tor_stackexchange",
            # future: exploit_db, nvd_nist, owasp_docs
        ],
    },
    "electronics": {
        "description": (
            "How do I wire this circuit? How do I use this microcontroller? "
            "How do I read a schematic? How do I program an Arduino? How do I "
            "use GPIO pins? What component do I need for this circuit? How do "
            "I measure voltage or current? How do I design a PCB? How do I "
            "debug this electronics problem? How do I use I2C or SPI? How do "
            "I set up amateur radio? How do I build a robot?"
        ),
        "patterns": [
            "electronics_stackexchange",
            "arduino_stackexchange",
            "ham_stackexchange",
            "robotics_stackexchange",
            "3dprinting_stackexchange",
            # future: datasheets, component_databases, kicad_docs
        ],
    },
    "reference": {
        "description": (
            "What is the history of this? Who was this person? What is this "
            "place? What does this word mean? What happened in this event? "
            "What is this scientific concept? What is the geography of this "
            "region? What is this cultural tradition? What is this theory? "
            "What is this philosophical idea? What is the etymology of this word?"
        ),
        "patterns": [
            "wikipedia",
            "wikibooks",
            "wikivoyage",
            "wikisource",
            "wiktionary",
            "history_stackexchange",
            "mythology_stackexchange",
            "literature_stackexchange",
            "philosophy_stackexchange",
            "linguistics_stackexchange",
            "internet_encyclopedia_philosophy",
            "ebooks_stackexchange",
            "parenting_stackexchange",
            "pets_stackexchange",
            "gis_stackexchange",
            "photo_stackexchange",
            "music_stackexchange",
            "openmusictheory",
            "libretexts_org_en_socialsci",
            "libretexts_org_en_human",
            # future: britannica, columbia, world_almanac
        ],
    },
}

# ── application state ─────────────────────────────────────────────────────────

app = Flask(__name__)
_embedder        = None
_client          = None         # shared chromadb.PersistentClient
_all_names       = []           # all permitted collection names
_col_cache       = {}           # name → {"col": Collection, "last_used": float}
_col_lock        = threading.Lock()
_group_cols      = {}           # group_name → [collection_name, ...]
_group_embs      = {}           # group_name → normalized np.ndarray (description embedding)
_args            = None


# ── group routing ─────────────────────────────────────────────────────────────

def _build_group_index(available_names: list[str]) -> None:
    """Assign available collections to groups and embed group descriptions."""
    global _group_cols, _group_embs

    assigned: set[str] = set()
    _group_cols = {}
    for gname, gdef in GROUPS.items():
        matched = [n for n in available_names if any(p in n for p in gdef["patterns"])]
        if matched:
            _group_cols[gname] = matched
            assigned.update(matched)

    unassigned = [n for n in available_names if n not in assigned]
    if unassigned:
        _group_cols["_other"] = unassigned

    named_groups = [g for g in _group_cols if g != "_other"]
    if named_groups:
        descs = [GROUPS[g]["description"] for g in named_groups]
        embs  = _embedder.encode(descs, normalize_embeddings=True)
        _group_embs = {g: embs[i] for i, g in enumerate(named_groups)}

    print("Collection groups:")
    for g, cols in _group_cols.items():
        label = g if g != "_other" else "other (unassigned)"
        print(f"  {label}: {len(cols)} collection(s)")


def _route_query(query_vec: np.ndarray) -> list[str]:
    """Return group names most relevant to the (normalized) query vector."""
    if not _group_embs:
        return list(_group_cols.keys())

    scores = {g: float(np.dot(query_vec, emb)) for g, emb in _group_embs.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best   = ranked[0][1] if ranked else 0.0

    if best < _args.route_threshold:
        # Low confidence: broaden search but still cap to avoid loading everything
        fallback = [g for g, _ in ranked[:_args.top_groups * 2]]
        if "_other" in _group_cols:
            fallback.append("_other")
        return fallback

    selected = [g for g, s in ranked[:_args.top_groups] if s >= best - 0.1]
    if "_other" in _group_cols:
        selected.append("_other")
    return selected


# ── collection cache ──────────────────────────────────────────────────────────

def _select_group_collections(names: list[str], query: str, max_n: int) -> list[str]:
    """When a group has more than max_n collections, pick the most name-relevant ones."""
    if len(names) <= max_n:
        return names
    words = {w for w in query.lower().split() if len(w) > 3}
    def name_score(n):
        nl = n.lower()
        return sum(1 for w in words if w in nl)
    return sorted(names, key=name_score, reverse=True)[:max_n]


def _ensure_loaded(names: list[str]) -> dict:
    """Lazy-load requested collections into cache; refresh last_used on each access."""
    now = time.time()
    with _col_lock:
        for n in names:
            if n not in _col_cache:
                # Evict LRU entry when cache is at capacity
                cap = _args.max_cache_size
                if cap and len(_col_cache) >= cap:
                    lru = min(_col_cache, key=lambda k: _col_cache[k]["last_used"])
                    del _col_cache[lru]
                _col_cache[n] = {"col": _client.get_collection(n), "last_used": now}
                print(f"  [cache] loaded: {n}", flush=True)
            else:
                _col_cache[n]["last_used"] = now
        return {n: _col_cache[n]["col"] for n in names if n in _col_cache}


def _eviction_loop() -> None:
    """Daemon thread: evict collections idle for longer than GROUP_TTL seconds."""
    while True:
        time.sleep(60)
        now = time.time()
        with _col_lock:
            stale = [n for n, e in _col_cache.items() if now - e["last_used"] > GROUP_TTL]
            for n in stale:
                del _col_cache[n]
            if stale:
                print(f"  [cache] evicted {len(stale)}: {', '.join(stale)}", flush=True)


# ── retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, k: int) -> list[dict]:
    q_norm = _embedder.encode([query], normalize_embeddings=True)
    q_vec  = q_norm.tolist()

    groups        = _route_query(q_norm[0])
    seen_names: set[str] = set()
    names_to_load = []
    for g in groups:
        selected = _select_group_collections(
            _group_cols.get(g, []), query, _args.max_per_group
        )
        for name in selected:
            if name not in seen_names:
                seen_names.add(name)
                names_to_load.append(name)

    col_map        = _ensure_loaded(names_to_load)
    cols_to_search = [col_map[n] for n in names_to_load if n in col_map]

    print(f"  groups: {[g for g in groups if g != '_other']} → {len(cols_to_search)} collections", flush=True)

    candidates = []
    for col in cols_to_search:
        try:
            results = col.query(
                query_embeddings=q_vec, n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"  skipping '{col.name}': {e}", flush=True)
            continue
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            candidates.append({
                "text": doc, "source": meta["source"],
                "title": meta["title"], "dist": dist,
                "is_accepted": meta.get("is_accepted", False),
                "zim": col.name.removesuffix("_chunks"),
            })

    # Boost accepted answers by reducing their distance score
    for c in candidates:
        if c.get("is_accepted"):
            c["dist"] *= 0.85
    candidates.sort(key=lambda c: c["dist"])
    seen, chunks = set(), []
    for c in candidates:
        key = (c["source"], c["text"][:80])
        if key not in seen:
            seen.add(key)
            chunks.append(c)
        if len(chunks) >= k:
            break
    return chunks


# ── prompt / streaming ────────────────────────────────────────────────────────

def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(f"[{c['title']}]\n{c['text']}" for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask")
def ask():
    question = request.args.get("q", "").strip()

    def generate():
        if not question:
            yield "data: [DONE]\n\n"
            return

        print(f"Q: {question}", flush=True)
        chunks = retrieve(question, _args.top_k)
        if not chunks:
            yield sse({"token": "No relevant content found in the index."})
            yield "data: [DONE]\n\n"
            return

        payload = {
            "model": _args.model,
            "system": SYSTEM_PROMPT,
            "prompt": build_prompt(question, chunks),
            "stream": True,
            "keep_alive": -1,
        }
        try:
            with requests.post(
                f"{_args.ollama_url}/api/generate",
                json=payload, stream=True, timeout=_args.timeout,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield sse({"token": token})
                    if chunk.get("done"):
                        break
        except requests.exceptions.ConnectionError:
            yield sse({"token": "\n[Error: could not reach Ollama — is it running?]"})
        except requests.exceptions.ReadTimeout:
            yield sse({"token": "\n[Error: Ollama timed out — the model may need more time on this hardware]"})
        except requests.exceptions.HTTPError as e:
            yield sse({"token": "\n[Error: the language model returned an error — please try again]"})
        except Exception as e:
            yield sse({"token": f"\n[Error: {e}]"})

        seen, sources = [], []
        for c in chunks:
            entry = {"title": c["title"], "source": c["source"], "zim": c.get("zim", "")}
            if entry not in seen:
                seen.append(entry)
                sources.append(entry)
        yield sse({"sources": sources})
        yield "data: [DONE]\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """Non-streaming JSON endpoint for automated testing and evaluation."""
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return {"error": "no question provided"}, 400

    t0 = time.time()
    print(f"Q (api): {question}", flush=True)
    chunks = retrieve(question, _args.top_k)
    if not chunks:
        return {"answer": "", "sources": [], "groups": [], "elapsed": round(time.time() - t0, 1),
                "error": "no relevant content found"}

    payload = {
        "model": _args.model,
        "system": SYSTEM_PROMPT,
        "prompt": build_prompt(question, chunks),
        "stream": False,
        "keep_alive": -1,
    }
    try:
        resp = requests.post(f"{_args.ollama_url}/api/generate", json=payload, timeout=_args.timeout)
        resp.raise_for_status()
        answer = resp.json().get("response", "")
    except Exception as e:
        return {"error": str(e)}, 500

    seen, sources = [], []
    for c in chunks:
        entry = {"title": c["title"], "source": c["source"]}
        if entry not in seen:
            seen.append(entry)
            sources.append(entry)

    return {
        "answer": answer,
        "sources": sources,
        "elapsed": round(time.time() - t0, 1),
    }


# ── startup ───────────────────────────────────────────────────────────────────

def main():
    global _embedder, _client, _all_names, _args

    parser = argparse.ArgumentParser(description="Kiwix RAG web interface.")
    parser.add_argument("--db", default=str(DB_PATH), help=f"ChromaDB path (default: {DB_PATH})")
    parser.add_argument("--collection", "-c", action="append", dest="collections",
                        metavar="NAME", help="Collection(s) to search (default: all)")
    parser.add_argument("--model", "-m", default=LLM_MODEL, help=f"Ollama model (default: {LLM_MODEL})")
    parser.add_argument("--embed-model", default=EMBED_MODEL,
                        help=f"Embedding model name or local path (default: {EMBED_MODEL})")
    parser.add_argument("--ollama-url", default=OLLAMA_URL, help=f"Ollama base URL (default: {OLLAMA_URL})")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"Chunks to retrieve (default: {TOP_K})")
    parser.add_argument("--top-groups", type=int, default=TOP_GROUPS,
                        help=f"Max groups to search per query (default: {TOP_GROUPS})")
    parser.add_argument("--route-threshold", type=float, default=ROUTE_THRESH,
                        help=f"Min similarity to select a group; below this searches all (default: {ROUTE_THRESH})")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Ollama request timeout in seconds (default: 300)")
    parser.add_argument("--max-collection-size", type=int, default=None, metavar="N",
                        help="Skip collections with more than N vectors")
    parser.add_argument("--max-per-group", type=int, default=15, metavar="N",
                        help="Max collections to search per group per query (default: 15)")
    parser.add_argument("--max-cache-size", type=int, default=15, metavar="N",
                        help="Max collections held in memory at once; evicts LRU (default: 15)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind host (default: 127.0.0.1; use 0.0.0.0 to serve the LAN)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    _args = parser.parse_args()

    print("Loading embedding model...", end=" ", flush=True)
    _embedder = SentenceTransformer(_args.embed_model)
    print("ready")

    db_path = Path(_args.db).expanduser()
    client  = chromadb.PersistentClient(path=str(db_path))
    available = [c.name for c in client.list_collections()]
    if not available:
        print("No collections found. Run build_index.py first.")
        sys.exit(1)

    names = _args.collections if _args.collections else available
    missing = [n for n in names if n not in available]
    if missing:
        print(f"Error: collection(s) not found: {', '.join(missing)}")
        sys.exit(1)

    if _args.max_collection_size is not None:
        conn = sqlite3.connect(str(db_path / "chroma.sqlite3"))
        rows = conn.execute(
            "SELECT c.name, COUNT(e.id) FROM collections c "
            "JOIN segments s ON s.collection = c.id "
            "JOIN embeddings e ON e.segment_id = s.id "
            "GROUP BY c.id"
        ).fetchall()
        conn.close()
        sizes = {name: count for name, count in rows}
        skipped = [n for n in names if not (0 < sizes.get(n, 0) <= _args.max_collection_size)]
        names   = [n for n in names if  0 < sizes.get(n, 0) <= _args.max_collection_size]
        if skipped:
            print(f"Skipped {len(skipped)} collection(s) (empty or >{_args.max_collection_size:,} vectors):")
            for n in skipped:
                print(f"  {sizes.get(n, 0):>10,}  {n}")

    _all_names = names
    _client    = client

    threading.Thread(target=_eviction_loop, daemon=True).start()

    print(f"\nBuilding group index over {len(names)} collections...")
    _build_group_index(names)

    print(f"\nModel: {_args.model} | top_k={_args.top_k} | top_groups={_args.top_groups}")
    print(f"Cache: max_cache_size={_args.max_cache_size} | max_per_group={_args.max_per_group}")
    print(f"Listening on http://{_args.host}:{_args.port}\n")
    app.run(host=_args.host, port=_args.port, threaded=True)


if __name__ == "__main__":
    main()
