#!/usr/bin/env python3
"""
Evaluation harness for the Kiwix RAG system.
Asks a curated set of questions via the /api/ask endpoint and scores
responses against expected keywords from the known source material.

Usage:
    python eval.py                          # test http://meshpi.local:5000
    python eval.py --url http://localhost:5000
    python eval.py --url http://meshpi.local:5000 --timeout 600
"""

import argparse
import json
import sys
import time

import requests

# ── test cases ────────────────────────────────────────────────────────────────
# expected_keywords: terms that MUST appear in a good answer given what we know
# is in the source material. Pass threshold is >= 50% of keywords present.
# Keep questions specific enough that a vague answer will fail.

TEST_CASES = [
    # ── medicine ──────────────────────────────────────────────────────────────
    {
        "group": "medicine",
        "question": "How do I treat a sucking chest wound?",
        "expected": ["seal", "dressing", "occlusive", "chest"],
        "note": "Tension pneumothorax / occlusive dressing — in zimgit_medicine and fas_military_medicine",
    },
    {
        "group": "medicine",
        "question": "What are the signs of a tension pneumothorax?",
        "expected": ["pressure", "lung", "breathing", "needle"],
        "note": "Tension pneumo signs/treatment — military medicine / health SE",
    },
    {
        "group": "medicine",
        "question": "What is the correct dose of ibuprofen for an adult?",
        "expected": ["mg", "dose", "day", "hours"],
        "note": "Basic dosing — NHS medicines",
    },
    {
        "group": "medicine",
        "question": "How do I apply a tourniquet correctly?",
        "expected": ["limb", "proximal", "tight", "windlass"],
        "note": "Tourniquet application — military medicine / zimgit_medicine",
    },

    # ── survival ──────────────────────────────────────────────────────────────
    {
        "group": "survival",
        "question": "How do I purify water in the wilderness?",
        "expected": ["boil", "filter", "chemical", "iodine"],
        "note": "Water purification — zimgit_water and outdoors SE",
    },
    {
        "group": "survival",
        "question": "What are the steps to tie a bowline knot?",
        "expected": ["loop", "end", "rope", "knot"],
        "note": "Bowline steps — zimgit_knots / outdoors SE",
    },
    {
        "group": "survival",
        "question": "What are non-electronic ways to signal for rescue if I am lost in the wilderness?",
        "expected": ["fire", "smoke", "mirror", "whistle"],
        "note": "Primitive distress signaling — outdoors SE / zimgit",
    },

    # ── coding ────────────────────────────────────────────────────────────────
    {
        "group": "coding",
        "question": "How do I undo a git commit without losing my changes?",
        "expected": ["reset", "revert", "commit", "undo"],
        "note": "git reset or revert — devdocs_en_git",
    },
    {
        "group": "coding",
        "question": "How do I reverse a list in Python?",
        "expected": ["reverse", "slice", "[::-1]", "list"],
        "note": "Basic Python — devdocs_en_python",
    },
    {
        "group": "coding",
        "question": "What is the difference between a pointer and a reference in C++?",
        "expected": ["pointer", "reference", "memory", "null"],
        "note": "C++ fundamentals — devdocs_en_cpp",
    },

    # ── devops ────────────────────────────────────────────────────────────────
    {
        "group": "devops",
        "question": "How do I expose a port in Docker?",
        "expected": ["EXPOSE", "publish", "-p", "host"],
        "note": "Docker port mapping — devdocs_en_docker",
    },
    {
        "group": "devops",
        "question": "How do I reload nginx configuration without downtime?",
        "expected": ["reload", "nginx", "configuration", "restart"],
        "note": "nginx reload — devdocs_en_nginx",
    },

    # ── web ───────────────────────────────────────────────────────────────────
    {
        "group": "web",
        "question": "How do I use the useEffect hook in React?",
        "expected": ["dependency", "array", "cleanup", "render"],
        "note": "React hooks — devdocs_en_react",
    },

    # ── cooking ───────────────────────────────────────────────────────────────
    {
        "group": "cooking",
        "question": "What is the Maillard reaction and why does it matter for cooking?",
        "expected": ["browning", "amino", "sugar", "flavor"],
        "note": "Cooking chemistry — cooking_stackexchange",
    },

    # ── automotive ────────────────────────────────────────────────────────────
    {
        "group": "automotive",
        "question": "What are the symptoms of a failing alternator?",
        "expected": ["battery", "light", "dim", "charge"],
        "note": "Alternator diagnosis — mechanics_stackexchange",
    },
    {
        "group": "aviation",
        "question": "What is the difference between flying VFR and IFR?",
        "expected": ["visual", "instrument", "cloud", "ceiling"],
        "note": "Pilot rules — aviation SE",
    },

    # ── mathematics ───────────────────────────────────────────────────────────
    {
        "group": "mathematics",
        "question": "What is the central limit theorem?",
        "expected": ["mean", "distribution", "sample", "normal"],
        "note": "Statistics — libretexts_stats",
    },

    # ── physics ───────────────────────────────────────────────────────────────
    {
        "group": "physics",
        "question": "Why does the sky appear blue?",
        "expected": ["scatter", "wavelength", "rayleigh", "light"],
        "note": "Atmospheric optics — physics SE / libretexts_phys",
    },

    # ── biology ───────────────────────────────────────────────────────────────
    {
        "group": "biology",
        "question": "What is the difference between DNA and RNA?",
        "expected": ["thymine", "uracil", "double", "ribose"],
        "note": "Molecular biology — biology SE",
    },

    # ── electronics ───────────────────────────────────────────────────────────
    {
        "group": "electronics",
        "question": "How do I debounce a button press in Arduino?",
        "expected": ["millis", "delay", "bounce", "state"],
        "note": "Arduino debounce — arduino SE",
    },
    {
        "group": "electronics",
        "question": "What causes layer delamination in FDM 3D printing and how do I fix it?",
        "expected": ["temperature", "layer", "cooling", "speed"],
        "note": "3D printing troubleshooting — 3dprinting SE",
    },

    # ── cooking (batch 2 content) ─────────────────────────────────────────────
    {
        "group": "cooking",
        "question": "What internal temperature should chicken reach to be food safe?",
        "expected": ["165", "internal", "temperature", "safe"],
        "note": "Food safety — usda_2015",
    },
    {
        "group": "cooking",
        "question": "What is the difference between an ale and a lager?",
        "expected": ["yeast", "fermentation", "bottom", "top"],
        "note": "Beer styles — beer SE",
    },

    # ── gardening ─────────────────────────────────────────────────────────────
    {
        "group": "gardening",
        "question": "How do I get rid of aphids on my plants without pesticides?",
        "expected": ["soap", "water", "spray", "neem"],
        "note": "Organic pest control — gardening SE",
    },

    # ── security ──────────────────────────────────────────────────────────────
    {
        "group": "security",
        "question": "How does Tor anonymize internet traffic?",
        "expected": ["relay", "onion", "encrypt", "circuit"],
        "note": "Tor architecture — tor SE",
    },

    # ── reference ─────────────────────────────────────────────────────────────
    {
        "group": "reference",
        "question": "What is the trolley problem and what does it illustrate in ethics?",
        "expected": ["utilitarian", "moral", "divert", "five"],
        "note": "Ethics — philosophy SE / internet-encyclopedia-philosophy",
    },
    {
        "group": "reference",
        "question": "What is the difference between syntax and semantics in linguistics?",
        "expected": ["meaning", "structure", "grammar", "sentence"],
        "note": "Linguistics — linguistics SE",
    },

    # ── data ──────────────────────────────────────────────────────────────────
    {
        "group": "data",
        "question": "What is the difference between a Type I and Type II error in hypothesis testing?",
        "expected": ["false positive", "false negative", "null", "power"],
        "note": "Statistics — libretexts_stats / scicomp SE",
    },

    # ── repair ────────────────────────────────────────────────────────────────
    {
        "group": "repair",
        "question": "How do I fix a stuck zipper?",
        "expected": ["teeth", "slider", "wax", "pencil"],
        "note": "Repair — crafts SE / ifixit",
    },
]


# ── runner ────────────────────────────────────────────────────────────────────

def ask(base_url: str, question: str, timeout: int) -> dict:
    resp = requests.post(
        f"{base_url}/api/ask",
        json={"question": question},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def score(answer: str, keywords: list[str]) -> tuple[list, list]:
    lower = answer.lower()
    hits   = [kw for kw in keywords if kw.lower() in lower]
    misses = [kw for kw in keywords if kw.lower() not in lower]
    return hits, misses


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Kiwix RAG system.")
    parser.add_argument("--url", default="http://localhost:5000", help="Base URL of the web server")
    parser.add_argument("--timeout", type=int, default=400, help="Request timeout in seconds")
    parser.add_argument("--group", help="Only run tests for this group")
    parser.add_argument("--json", action="store_true", dest="json_out", help="Output results as JSON")
    args = parser.parse_args()

    cases = [t for t in TEST_CASES if not args.group or t["group"] == args.group]
    if not cases:
        print(f"No test cases for group '{args.group}'")
        sys.exit(1)

    results = []
    passed  = 0
    total_time = 0.0

    if not args.json_out:
        print(f"Evaluating {args.url}")
        print(f"Running {len(cases)} test(s)\n{'=' * 64}")

    for i, tc in enumerate(cases, 1):
        if not args.json_out:
            print(f"\n[{i}/{len(cases)}] [{tc['group'].upper()}] {tc['question']}")

        try:
            result = ask(args.url, tc["question"], args.timeout)
        except Exception as e:
            if not args.json_out:
                print(f"  ERROR: {e}")
            results.append({"question": tc["question"], "group": tc["group"],
                            "error": str(e), "passed": False})
            continue

        answer  = result.get("answer", "")
        elapsed = result.get("elapsed", 0)
        sources = result.get("sources", [])
        hits, misses = score(answer, tc["expected"])
        ok = len(hits) / len(tc["expected"]) >= 0.5
        if ok:
            passed += 1
        total_time += elapsed

        rec = {
            "question": tc["question"],
            "group": tc["group"],
            "passed": ok,
            "hits": hits,
            "misses": misses,
            "elapsed": elapsed,
            "sources": [s["title"] for s in sources],
            "answer": answer,
        }
        results.append(rec)

        if not args.json_out:
            print(f"  Time:    {elapsed}s")
            print(f"  Sources: {[s['title'] for s in sources]}")
            print(f"  Hits:    {hits}")
            if misses:
                print(f"  Misses:  {misses}")
            # Print up to 300 chars of answer
            snippet = answer.replace("\n", " ").strip()
            print(f"  Answer:  {snippet[:300]}{'...' if len(snippet) > 300 else ''}")
            status = "✓ PASS" if ok else "✗ FAIL"
            print(f"  {status}  ({len(hits)}/{len(tc['expected'])} keywords)")

    if args.json_out:
        print(json.dumps(results, indent=2))
    else:
        avg = total_time / len(cases) if cases else 0
        print(f"\n{'=' * 64}")
        print(f"Results:  {passed}/{len(cases)} passed")
        print(f"Avg time: {avg:.1f}s per question")

        if passed < len(cases):
            print("\nFailed questions:")
            for r in results:
                if not r.get("passed"):
                    print(f"  [{r['group']}] {r['question']}")


if __name__ == "__main__":
    main()
