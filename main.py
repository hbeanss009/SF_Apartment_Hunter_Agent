"""
Apartment Hunter Agent — main entrypoint.

User input (search description and refinement text) is parsed by an LLM (Google Gemini)
into structured criteria. The LLM extracts: neighbourhoods, min_bedrooms, min_bathrooms,
budget_range, must_have_amenities, nice_to_have_amenities. These are then used by
discovery → analyse → recommend.
"""
import json
import os
import re
import time
import warnings
from typing import Optional, Tuple

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from discovery import run_discovery
from Analyse import get_matching_apartments
from recommend import recommend_apartments

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

app = Flask(__name__)

# Store last analyse output and preferences so "run recommend again" can re-run only recommend.py
_last_analyse_output: Optional[str] = None
_last_user_preferences: Optional[list] = None
# Store last search description so "refine" can re-run pipeline with additional criteria
_last_description: Optional[str] = None

# Order of fields in user_preferences list (for merging refined criteria)
_PREFERENCE_KEYS = [
    "neighbourhoods",
    "min_bedrooms",
    "min_bathrooms",
    "budget_range",
    "must_have_amenities",
    "nice_to_have_amenities",
]


def _try_parse_json_from_llm(s: str) -> Optional[dict]:
    """Parse JSON from LLM output; strip markdown code blocks and extract first {...} if needed."""
    s = (s or "").strip()
    # Strip ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", s)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def _parse_additional_criteria(additional_text: str) -> Optional[dict]:
    """
    Use the LLM to extract only the criteria the user is updating in their additional message
    (budget, bedrooms, bathrooms, neighbourhood, amenities).
    Returns a dict with only the keys that were explicitly mentioned; omit keys
    not mentioned so we don't overwrite with empty.
    """
    api_key = _get_google_api_key()
    if not api_key or not api_key.strip():
        return None

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You extract apartment search criteria from a short "additional criteria" message.
The user already had a previous search; they are now adding or changing only some criteria.

Output ONLY a single valid JSON object. No markdown, no code block, no explanation.
Include ONLY the keys that the user explicitly mentioned or changed in this message.
Omit any key they did not mention (so we do not overwrite their previous value with empty).
Valid keys (include only if mentioned): neighbourhoods, min_bedrooms, min_bathrooms, budget_range, must_have_amenities, nice_to_have_amenities.
- neighbourhoods: string (comma-separated neighbourhood names)
- min_bedrooms: integer (minimum number of bedrooms)
- min_bathrooms: integer (minimum number of bathrooms)
- budget_range: string (e.g. "2000-4000")
- must_have_amenities: string (comma-separated)
- nice_to_have_amenities: string (comma-separated)
""",
            ),
            (
                "human",
                "Additional criteria from user:\n{additional}\n\nExtract only the fields they mentioned as JSON.",
            ),
        ]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key.strip(),
        temperature=0,
    )
    chain = prompt | llm
    try:
        raw = chain.invoke({"additional": additional_text})
    except Exception as e:
        print(f"[LLM parse additional] API call failed: {e}")
        return None
    text = raw.content if hasattr(raw, "content") else str(raw)
    data = _try_parse_json_from_llm(text or "")
    if not isinstance(data, dict):
        return None
    # Normalize to string values and drop keys not in our schema
    allowed = set(_PREFERENCE_KEYS)
    updates = {}
    for k, v in data.items():
        if k in allowed and v is not None:
            val = str(v).strip()
            if val:
                updates[k] = val
    return updates if updates else None


def _merge_preferences(previous: list[str], updates: dict) -> list[str]:
    """Merge updates into previous preferences; preference order is _PREFERENCE_KEYS."""
    previous_copy = list(previous) if len(previous) >= 6 else (previous + [""] * 6)[:6]
    for i, key in enumerate(_PREFERENCE_KEYS):
        if key in updates:
            previous_copy[i] = updates[key]
    return previous_copy


def _get_google_api_key() -> Optional[str]:
    """Return Gemini/Google API key from GOOGLE_API_KEY or GEMINI_API_KEY."""
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def _parse_user_preferences(description: str) -> Optional[list[str]]:
    """
    Use the LLM (Gemini) to parse the user's freeform description into structured
    apartment search criteria. The LLM extracts neighbourhoods, bedrooms, bathrooms,
    budget, and amenities so the rest of the pipeline can use them.

    Returns [neighbourhoods, min_bedrooms, min_bathrooms, budget_range,
             must_have_amenities, nice_to_have_amenities] or None on failure.
    """
    api_key = _get_google_api_key()
    if not api_key or not api_key.strip():
        print("[LLM parse] No API key: set GOOGLE_API_KEY or GEMINI_API_KEY in .env")
        return None

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You extract structured apartment search criteria from a short description.

Output ONLY a single valid JSON object. No markdown, no code block, no explanation.
Use these exact keys:
- neighbourhoods: string (comma-separated neighbourhood/area names)
- min_bedrooms: integer (minimum number of bedrooms)
- min_bathrooms: integer (minimum number of bathrooms)
- budget_range: string (e.g. "2000-4000" for monthly rent)
- must_have_amenities: string (comma-separated must-have amenities)
- nice_to_have_amenities: string (comma-separated nice-to-have amenities)

If a field is not specified, infer from context or use empty string. Always include all six keys.
""",
            ),
            (
                "human",
                "User description:\n{description}\n\nExtract the criteria as JSON.",
            ),
        ]
    )

    for model_name in ("gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"):
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key.strip(),
            temperature=0,
        )
        chain = prompt | llm
        try:
            raw = chain.invoke({"description": description})
        except Exception as e:
            print(f"[LLM parse] API call failed with {model_name}: {e}")
            continue
        text = raw.content if hasattr(raw, "content") else str(raw)
        if not (text and text.strip()):
            print(f"[LLM parse] {model_name} returned empty response.")
            continue
        data = _try_parse_json_from_llm(text)
        if isinstance(data, dict):
            break
        print(f"[LLM parse] {model_name} output was not valid JSON. Raw (first 500 chars): {repr((text or '')[:500])}")
    else:
        print("[LLM parse] All models failed or returned unparseable JSON.")
        return None

    neighbourhoods = str(data.get("neighbourhoods", "") or "")
    min_bedrooms = str(data.get("min_bedrooms", "") or "")
    min_bathrooms = str(data.get("min_bathrooms", "") or "")
    budget_range = str(data.get("budget_range", "") or "")
    must_have_amenities = str(data.get("must_have_amenities", "") or "")
    nice_to_have_amenities = str(data.get("nice_to_have_amenities", "") or "")

    return [
        neighbourhoods,
        min_bedrooms,
        min_bathrooms,
        budget_range,
        must_have_amenities,
        nice_to_have_amenities,
    ]


def run_pipeline(description: str, skip_discovery: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """
    Run full pipeline: parse description -> [discovery if not skipped] -> analyse -> recommend.
    Returns (recommendations_text or None, error_message or None).
    """
    user_preferences = _parse_user_preferences(description)
    if user_preferences is None:
        return (
            None,
            "The LLM could not parse your criteria. In .env set GOOGLE_API_KEY or GEMINI_API_KEY (get one at "
            "https://aistudio.google.com/apikey). Ensure your description includes neighbourhoods, "
            "min bedrooms/bathrooms, budget range, and amenities. Check the server console for details.",
        )

    if not skip_discovery:
        try:
            run_discovery()
        except Exception as e:
            return (None, f"Discovery failed: {e}")

    try:
        analyse_output = get_matching_apartments(user_preferences)
    except Exception as e:
        return (None, f"Match step failed: {e}")

    global _last_analyse_output, _last_user_preferences
    _last_analyse_output = analyse_output or ""
    _last_user_preferences = user_preferences

    if analyse_output and analyse_output.strip():
        with open("apartment_results.txt", "w", encoding="utf-8") as f:
            f.write(analyse_output)

    try:
        recommendations = recommend_apartments(analyse_output or "", user_preferences)
    except Exception as e:
        return (None, f"Recommendation step failed: {e}")

    if recommendations and recommendations.strip():
        with open("apartment_results.txt", "w", encoding="utf-8") as f:
            f.write(recommendations)
    return (recommendations or "", None)


@app.route("/")
def index():
    """Serve the apartment search form."""
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    """
    Run pipeline: discovery -> analyse -> recommend.
    Expects JSON body: { "description": "user's apartment requirements" }.
    Returns JSON: { "recommendations": "..." } or { "error": "..." }.
    """
    data = request.get_json(silent=True) or {}
    description = (data.get("description") or "").strip()
    if not description:
        return jsonify({"error": "Please describe the kind of apartment you're looking for."}), 400

    global _last_description
    _last_description = description
    recommendations, error = run_pipeline(description, skip_discovery=False)
    if error:
        return jsonify({"error": error}), 200
    return jsonify({"recommendations": recommendations or "No matching apartments were found."})


@app.route("/recommend", methods=["POST"])
def run_recommend_only():
    """
    Run only recommend.py again using the last analyse output and user preferences
    from the most recent full search. No discovery or analyse step.
    """
    global _last_analyse_output, _last_user_preferences
    if _last_analyse_output is None or _last_user_preferences is None:
        return jsonify({
            "error": "No previous search to re-rank. Run a full search first (Find Apartments)."
        }), 200
    try:
        recommendations = recommend_apartments(_last_analyse_output, _last_user_preferences)
        if recommendations and recommendations.strip():
            with open("apartment_results.txt", "w", encoding="utf-8") as f:
                f.write(recommendations)
        return jsonify({"recommendations": recommendations or "No recommendations returned."})
    except Exception as e:
        return jsonify({"error": f"Recommendation failed: {e}"}), 200


@app.route("/refine", methods=["POST"])
def refine_search():
    """
    Re-run the pipeline with additional criteria (no new discovery/scraping).
    Expects JSON: { "additional_criteria": "..." }.
    If the user adds or changes budget, bedrooms, bathrooms, neighbourhood, or amenities,
    we parse those updates, merge with the last preferences, and re-run analyse + recommend.
    Otherwise we combine with _last_description and re-parse the full query.
    """
    global _last_description, _last_analyse_output, _last_user_preferences
    if not _last_description:
        return jsonify({
            "error": "No previous search to refine. Run a full search first (Find Apartments)."
        }), 200
    data = request.get_json(silent=True) or {}
    additional = (data.get("additional_criteria") or "").strip()
    if not additional:
        return jsonify({"error": "Please describe what additional criteria you're looking for."}), 400

    # Try to parse additional criteria as structured updates (budget, bedrooms, bathrooms, neighbourhood, amenities)
    updates = _parse_additional_criteria(additional)
    if updates and _last_user_preferences is not None:
        # Merge updated criteria with previous preferences and re-run analyse + recommend
        merged_preferences = _merge_preferences(_last_user_preferences, updates)
        _last_description = f"{_last_description}. Additional criteria: {additional}"
        _last_user_preferences = merged_preferences
        try:
            analyse_output = get_matching_apartments(merged_preferences)
            _last_analyse_output = analyse_output or ""
            if analyse_output and analyse_output.strip():
                with open("apartment_results.txt", "w", encoding="utf-8") as f:
                    f.write(analyse_output)
            recommendations = recommend_apartments(analyse_output or "", merged_preferences)
            if recommendations and recommendations.strip():
                with open("apartment_results.txt", "w", encoding="utf-8") as f:
                    f.write(recommendations)
            return jsonify({"recommendations": recommendations or "No matching apartments were found."})
        except Exception as e:
            return jsonify({"error": f"Refine failed: {e}"}), 200

    # Fallback: combine description and re-run full pipeline (re-parse everything)
    combined = f"{_last_description}. Additional criteria: {additional}"
    _last_description = combined
    recommendations, error = run_pipeline(combined, skip_discovery=True)
    if error:
        return jsonify({"error": error}), 200
    return jsonify({"recommendations": recommendations or "No matching apartments were found."})


def main_cli() -> None:
    """
    CLI fallback: ask for input, run pipeline, print results.
    """
    print(
        "Tell me about the apartment you're looking for.\n"
        "For best results, include:\n"
        "- neighbourhoods you like\n"
        "- minimum bedrooms and bathrooms\n"
        "- your monthly budget range\n"
        "- must-have amenities\n"
        "- nice-to-have amenities\n"
    )
    description = input("Describe your ideal apartment in a few sentences: ")
    recommendations, error = run_pipeline(description, skip_discovery=False)
    if error:
        print(f"Error: {error}")
    else:
        print("\nRecommended apartments:\n")
        print(recommendations or "No matching apartments were found.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)

