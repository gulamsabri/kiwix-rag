# kiwix_rag/groups.py
"""
GROUPS — semantic routing table used by GroupRouter.
SYSTEM_PROMPT — injected into every LLM request.

To add a new ZIM collection: add its collection-name substring to the
appropriate group's "patterns" list. Patterns match against collection
names that use underscores throughout (dots and hyphens are converted
during indexing). Example: use "health_stackexchange" not
"health.stackexchange".
"""

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
            "survivorlibrary_medicine",
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
            "survivorlibrary_agriculture",
            "survivorlibrary_homesteading",
            "s2underground",
            # future: army_survival_fm, sere
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
            "survivorlibrary_agriculture",
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
            "survivorlibrary_homesteading",
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
            "survivorlibrary_engineering",
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
            "survivorlibrary_amateur_radio",
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
            "survivorlibrary_reference",
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
