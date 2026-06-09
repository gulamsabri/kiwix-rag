#!/usr/bin/env python3
"""
Re-segment survivorlibrary ChromaDB collections into topic-focused sub-collections.

Auto-detects all survivorlibrary_* collections that aren't the six topic targets,
classifies each chunk by its source filename, and writes into the topic collections
without re-embedding. Safe to run after each new batch that includes survivorlibrary.

New collections:
  survivorlibrary_amateur_radio  — 73 Magazine (all issues)
  survivorlibrary_medicine       — Medical reference books
  survivorlibrary_agriculture    — Farming, botany, livestock, beekeeping, veterinary
  survivorlibrary_engineering    — Industrial arts, chemistry, machinery, manufacturing
  survivorlibrary_homesteading   — Cooking, food preservation, domestic arts, crafts
  survivorlibrary_reference      — History, law, astronomy, art, and everything else

Usage:
    python resegment_survivorlibrary.py --dry-run     # show category counts, no writes
    python resegment_survivorlibrary.py               # build new collections
    python resegment_survivorlibrary.py --drop-old    # also delete the source collections
"""

import argparse
import sys
from pathlib import Path

import chromadb

TARGET_PREFIX = "survivorlibrary"
CATEGORIES = ["amateur_radio", "medicine", "agriculture", "engineering", "homesteading", "reference"]

# Collections to never treat as sources (the topic collections themselves)
_TARGET_NAMES = {f"{TARGET_PREFIX}_{cat}" for cat in CATEGORIES}


def find_source_collections(client) -> list[str]:
    """Return survivorlibrary batch collections (name contains _chunks) that aren't topic targets."""
    return sorted(
        c.name for c in client.list_collections()
        if "survivorlibrary" in c.name
        and "_chunks" in c.name
        and c.name not in _TARGET_NAMES
    )

FETCH_BATCH = 200   # small because we're fetching embedding vectors

# --- Classifier ---

# Exact matches on the last path component (directory-level category pages)
_DIRECTORY_MAP = {
    # Medicine
    "medical_diagnostics": "medicine",
    "medical_courses_us_army": "medicine",
    "medical_emergency": "medicine",
    "medical_surgery_1": "medicine",
    "medical_surgery_2": "medicine",
    "medical_surgery_1900-1922": "medicine",
    "medical_medicine_1900-1922": "medicine",
    "medical_obstetrics_1900-1922": "medicine",
    "medical_anesthesia": "medicine",
    "medical_hypnotism": "medicine",
    "medical_microscopy": "medicine",
    "medical_nursing": "medicine",
    "medical_x_rays": "medicine",
    "nbc": "medicine",
    "dentistry": "medicine",
    "sanitation": "medicine",
    "opium": "medicine",
    # Agriculture
    "farming": "agriculture",
    "farming_potato_and_sweet_potato": "agriculture",
    "farming_corn": "agriculture",
    "farming_fish": "agriculture",
    "beekeeping": "agriculture",
    "bee_journal_british": "agriculture",
    "bee_journal_american": "agriculture",
    "veterinary": "agriculture",
    "horses": "agriculture",
    "dogs": "agriculture",
    "livestock_cattle": "agriculture",
    "livestock_swine": "agriculture",
    "livestock_sheep": "agriculture",
    "livestock_rabbits_and_cavies": "agriculture",
    "poultry": "agriculture",
    "botany": "agriculture",
    "forestry": "agriculture",
    "trapping_and_hunting": "agriculture",
    "mushrooms": "agriculture",
    "berries": "agriculture",
    "boy_scout_manuals": "agriculture",
    "archery": "agriculture",
    "fishing": "agriculture",
    "grapes_wine_raisins": "agriculture",
    "tobacco": "agriculture",
    "rat_control": "agriculture",
    "survival_individual": "agriculture",
    "herbalism": "agriculture",
    # Engineering
    "engineering_electrical": "engineering",
    "engineering_general": "engineering",
    "engineering_hydraulics": "engineering",
    "engineering_drainage": "engineering",
    "coal_and_mining": "engineering",
    "fuels": "engineering",
    "steam_engines": "engineering",
    "heavy_industrial_machinery": "engineering",
    "machine_tools": "engineering",
    "machinerys_reference": "engineering",
    "forging_and_casting": "engineering",
    "smithing": "engineering",
    "welding": "engineering",
    "drilling": "engineering",
    "boilermaker": "engineering",
    "concrete": "engineering",
    "stone_and_masonry": "engineering",
    "bridges_and_dams": "engineering",
    "railroads": "engineering",
    "aeroplanes": "engineering",
    "airships": "engineering",
    "shipbuilding": "engineering",
    "telegraph_and_telephone": "engineering",
    "radio": "engineering",
    "photography": "engineering",
    "glassmaking": "engineering",
    "pottery": "engineering",
    "printing": "engineering",
    "papermaking": "engineering",
    "leather": "engineering",
    "cotton": "engineering",
    "hemp_and_flax": "engineering",
    "silk_culture": "engineering",
    "turpentine_glue_solvents": "engineering",
    "gunpowder_and_explosives": "engineering",
    "firearms_manuals": "engineering",
    "firearms_books": "engineering",
    "wind_and_water": "engineering",
    "chemistry": "engineering",
    "architecture": "engineering",
    "construction": "engineering",
    "geodesy": "engineering",
    "surveying": "engineering",
    "navigation": "engineering",
    "meteorology": "engineering",
    "mechanical_drawing": "engineering",
    "clockmaking": "engineering",
    "scientific_american_series_1": "engineering",
    "scientific_american_series_2": "engineering",
    "formulas": "engineering",
    "sliderules": "engineering",
    "sewage": "engineering",
    "heating": "engineering",
    "refrigeration": "engineering",
    "engraving_and_woodcuts": "engineering",
    # Homesteading
    "canning": "homesteading",
    "cooking_and_cookbooks": "homesteading",
    "baking": "homesteading",
    "cheese_and_butter": "homesteading",
    "food": "homesteading",
    "brewing_and_distilling": "homesteading",
    "home_economics": "homesteading",
    "sewing": "homesteading",
    "knitting_lace_needlepoint": "homesteading",
    "weaving": "homesteading",
    "wood_furniture": "homesteading",
    "wood_carpentry": "homesteading",
    "wood_carving": "homesteading",
    "bookbinding": "homesteading",
    "hatmaking": "homesteading",
    "shoemaking": "homesteading",
    "basketry": "homesteading",
    "shelter": "homesteading",
    "coffee_and_tea": "homesteading",
    "butchering": "homesteading",
    # Amateur radio
    "radio_73_magazine": "amateur_radio",
}

# Keyword patterns checked against the lowercased filename stem (order = priority)
_FILENAME_RULES: list[tuple[str, list[str]]] = [
    ("amateur_radio", ["73_magazine"]),
    ("medicine", [
        "medicine", "materia_medica", "surgical", "surgery", "anatomy", "physiology",
        "clinical", "obstetric", "gynecolog", "nursing", "diagnosis", "diagnostic",
        "hypnotism", "microscop", "x_ray", "domestic_medicine", "nature_cures",
        "parasitology", "anesthesia", "pharmacop", "compend_of_operative",
        "course_in_surgical", "chapter_in_minor_surgery", "childs_book_of_the_teeth",
        "handbook_of_physical_diagnosis", "compendium_of_obstetrics",
        "handbook_of_physical", "guide_to_the_dissection",
        "handbook_of_practical_parasitology",
    ]),
    ("agriculture", [
        "farm", "garden", "agricult", "livestock", "cattle", "swine", "sheep",
        "poultry", "horse", "botany", "beekeep", "apiculture", "veterinary",
        "strawberry", "grape", "wine", "herb", "mushroom", "forest",
        "corn", "potato", "plant_lore", "fishing", "big_game", "hunting",
        "trapping", "rat_control", "dog", "bee_journal", "bee_", "berries",
        "boy_scout", "angling", "guide_to_the_dog", "percheron",
        "description_of_the_collie", "book_about_bees", "garden_of_herbs",
        "handbook_of_angling", "compendium_of_botany", "course_of_practical_instruction_in_botany",
        "handbook_of_systematic_botany", "course_of_practical_instruction_in_botany",
        "condensed_botany", "comparative_study_of_winter", "tobacco",
        "description_of_ceylon", "coffee", "complete_manual_for_the_cultivation",
    ]),
    ("engineering", [
        "steam_engine", "steam-engine", "catechism_of_the_steam", "handbook_on_the_steam",
        "chemistry", "chemical", "engineer", "mechanic", "mining", "machinery",
        "electric", "calico", "dyeing", "cotton", "leather", "glassmaking",
        "pottery", "welding", "gunpowder", "explosive", "locomotive",
        "boiler", "handbook_of_practical_gas", "cement", "concrete",
        "hydraulic", "telegraph", "telephone", "photography", "clockmaking",
        "planing_mill", "turpentine", "glue", "solvent", "papermaking",
        "printing", "bookbinding", "lithography", "laboratory_glass",
        "foundry", "smithing", "forging", "casting", "machine_tool",
        "drill", "stone_and_masonry", "wind_and_water", "aeroplanes",
        "airships", "shipbuilding", "bridges", "dams", "refrigeration",
        "heating", "surveying", "geodesy", "navigation", "barometer",
        "sewage", "drainage", "scientific_american", "formulas", "slide_rule",
        "sliderule", "compendium_of_mechanics", "firearms", "rifle_musket",
        "industrial", "manufacture", "construction", "architectural",
        "railroads", "locomotive_valve", "silk_culture", "hemp", "flax",
        "descriptive_treatise_on_mining", "formula_for_the_flow_of_water",
        "handbook_for_cement", "handbook_of_laboratory_glass",
        "history_of_the_growth_of_the_steam",
        "history_and_description_of_modern_wine",  # wine-making = engineering here? no → agriculture
        "75_years_of_gas", "boys_text_book_on_gas", "handbook_of_practical_gas",
        "history_of_the_planing_mill", "compendium_of_mechanics",
        "few_useful_shop_hints", "course_in_mechanical_drawing",
        "course_in_structural_drafting", "course_in_elementary_mechanical_drawing",
        "course_in_the_principles_of_mechanical_drawing",
        "complete_handbook_of_tailoring",  # tailoring = homesteading actually
        "description_of_a_clepsydra",
        "descriptive_catalogue_of_manufactures",
        "dictionary_of_calico_printing",
        "dictionary_of_practical_and_theoretical_chemistry",
        "dictionary_of_the_art_of_printing",
        "dictionary_of_electrical_words",
        "handbook_of_industrial_organic_chemistry",
        "discussion_of_the_prevailing_theories_and_practices_relating_to_sewage",
        "digest_of_facts_relating_to_the_treatment_and_utilization_of_sewage",
        "handbook_of_sewage_utilization",
        "history_of_the_growth_of_the_steam-engine",
        "handbook_of_forest_protection",
    ]),
    ("homesteading", [
        "household", "cooking", "cook", "baking", "canning", "preservation",
        "soap", "cheese", "butter", "sewing", "tailoring", "dressmaking",
        "laundry", "knitting", "basketry", "home_economics", "homemaker",
        "shell_fish", "recipes", "food", "brewing", "distilling",
        "spinning", "weaving", "pottery_painting", "leather_work", "wood_carving",
        "shelter", "butchering", "a_b_c_in_cheese", "abc_in_cheese",
        "course_of_study_for_homemakers", "course_in_household_arts",
        "domestic_cyclopedia", "domestic_cyclopedia_of_practical",
        "hand-book_of_house_sanitation", "handbook_for_sewing",
        "handbook_of_elementary_sewing", "complete_course_in_dressmaking",
        "complete_handbook_of_tailoring", "complete_guide_to_ornamental_leather",
        "300_ways_to_cook", "175_choice_recipes",
        "comprehensive_view_of_the_culture_of_the_vine",
        "history_and_description_of_modern_wine",
        "history_and_description_of_modern_wines",
        "brief_discourse_on_wine",
        "cup_of_coffee",
        "complete_guide_to_spinning",
        "complete_system_of_cutting",
        "course_in_slide_rules",  # slide rules → actually engineering
        "a-complete-handbook-of-nature-cures",  # → medicine actually
    ]),
]


def classify(source: str) -> str:
    """Return the category name for a chunk's source path."""
    parts = [p.lower().replace(".pdf", "").replace(".html", "") for p in source.split("/") if p]

    # Empty or navigation junk → reference
    if not parts:
        return "reference"
    name = parts[-1]
    if not name or name.startswith("?") or name.startswith("index.php"):
        return "reference"

    # Check every path component against the directory map — the key is usually a
    # directory name like "farming", not the filename inside it.
    for part in parts:
        if part in _DIRECTORY_MAP:
            return _DIRECTORY_MAP[part]

    # Keyword scan on the filename in priority order
    for category, keywords in _FILENAME_RULES:
        if any(kw in name for kw in keywords):
            return category

    return "reference"


# --- Main ---

def build_collections(client, source_cols, dry_run: bool) -> dict[str, int]:
    """Stream all chunks, classify, and populate target collections. Returns counts."""
    target_name = {cat: f"{TARGET_PREFIX}_{cat}" for cat in CATEGORIES}
    counts: dict[str, int] = {cat: 0 for cat in CATEGORIES}

    if not dry_run:
        targets = {
            cat: client.get_or_create_collection(
                target_name[cat], metadata={"hnsw:space": "cosine"}
            )
            for cat in CATEGORIES
        }
        # Buffer per category so we can add in batches
        buffers: dict[str, dict] = {
            cat: {"ids": [], "embeddings": [], "documents": [], "metadatas": []}
            for cat in CATEGORIES
        }

    for src_name in source_cols:
        try:
            src = client.get_collection(src_name)
        except Exception:
            print(f"  WARNING: source collection '{src_name}' not found, skipping.")
            continue

        total = src.count()
        print(f"\n  {src_name}  ({total:,} chunks)")
        offset = 0
        fetched = 0

        while offset < total:
            result = src.get(
                limit=FETCH_BATCH,
                offset=offset,
                include=["embeddings", "documents", "metadatas"],
            )
            if not result["ids"]:
                break

            for chunk_id, emb, doc, meta in zip(
                result["ids"], result["embeddings"],
                result["documents"], result["metadatas"]
            ):
                source = (meta or {}).get("source", "")
                cat = classify(source)
                counts[cat] += 1

                if not dry_run:
                    buf = buffers[cat]
                    # Prefix the ID with the source collection to guarantee uniqueness
                    buf["ids"].append(f"{src_name}:{chunk_id}")
                    buf["embeddings"].append(emb)
                    buf["documents"].append(doc)
                    buf["metadatas"].append(meta)

                    if len(buf["ids"]) >= 500:
                        targets[cat].upsert(**buf)
                        buf["ids"].clear(); buf["embeddings"].clear()
                        buf["documents"].clear(); buf["metadatas"].clear()

            fetched += len(result["ids"])
            offset += len(result["ids"])
            print(f"\r    {fetched:,} / {total:,}", end="", flush=True)

        print()

    if not dry_run:
        # Flush remaining buffers
        for cat, buf in buffers.items():
            if buf["ids"]:
                targets[cat].upsert(**buf)

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Re-segment survivorlibrary collections by topic."
    )
    parser.add_argument("--db", default=str(Path(__file__).parent / "vector_db"),
                        help="ChromaDB path (default: ./vector_db)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Classify and count without writing any collections")
    parser.add_argument("--drop-old", action="store_true",
                        help="Delete the two source collections after successful build")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.db)

    source_cols = find_source_collections(client)
    if not source_cols:
        print("No survivorlibrary source collections found (nothing to segment).", file=sys.stderr)
        sys.exit(0)

    # Warn if targets already exist
    existing = {c.name for c in client.list_collections()}
    target_names = [f"{TARGET_PREFIX}_{cat}" for cat in CATEGORIES]
    existing_targets = [n for n in target_names if n in existing]
    if existing_targets and not args.dry_run:
        print("NOTE: the following target collections already exist; new chunks will be upserted:")
        for n in existing_targets:
            print(f"  {n}")
        print("Use --drop-old after a clean run if you want to rebuild from scratch.\n")

    # Snapshot pre-existing target counts so verification can be accurate on reruns
    pre_existing: dict[str, int] = {}
    if not args.dry_run:
        for cat in CATEGORIES:
            name = f"{TARGET_PREFIX}_{cat}"
            try:
                pre_existing[cat] = client.get_collection(name).count()
            except Exception:
                pre_existing[cat] = 0

    mode = "DRY RUN" if args.dry_run else "LIVE BUILD"
    print(f"Mode:    {mode}")
    print(f"DB:      {args.db}")
    print(f"Sources: {source_cols}")
    print()

    counts = build_collections(client, source_cols, args.dry_run)

    total = sum(counts.values())
    print(f"\n{'='*60}")
    print(f"{'Category':<25}  {'Chunks':>8}  {'%':>5}  Target collection")
    print(f"{'-'*60}")
    for cat in CATEGORIES:
        n = counts[cat]
        pct = 100 * n / total if total else 0
        print(f"  {cat:<23}  {n:>8,}  {pct:>4.1f}%  {TARGET_PREFIX}_{cat}")
    print(f"  {'TOTAL':<23}  {total:>8,}")

    if args.dry_run:
        print("\n[dry-run] No collections written. Re-run without --dry-run to build.")
    else:
        print("\nVerifying collection counts...")
        ok = True
        for cat in CATEGORIES:
            col = client.get_collection(f"{TARGET_PREFIX}_{cat}")
            actual = col.count()
            expected = pre_existing[cat] + counts[cat]
            status = "✓" if actual == expected else f"MISMATCH (expected {expected})"
            print(f"  {TARGET_PREFIX}_{cat}: {actual:,}  {status}")
            if actual != expected:
                ok = False

        if ok and args.drop_old:
            print("\nDropping source collections...")
            for name in source_cols:
                client.delete_collection(name)
                print(f"  deleted {name}")
        elif not ok:
            print("\nWARNING: count mismatches detected — source collections NOT deleted.")


if __name__ == "__main__":
    main()
