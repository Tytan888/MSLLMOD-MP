import os
import json
import time
import argparse
import traceback

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from tqdm import tqdm

import google.generativeai as genai

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your environment.")
genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"

CRITERIA = [
    ("Accuracy", 1.5),
    ("Fluency", 1.5),
    ("Completeness", 0.5),
    ("Cultural Appropriateness", 0.5),
]

PROMPT_TEMPLATE = """
You are a translation judge specialized in English-to-Filipino (Tagalog) evaluation.

Given:
- SOURCE (English)
- TRANSLATION (Filipino)

Evaluate the TRANSLATION using the following criteria. For each criterion, give:
- A score from 1 (worst) to 5 (best)
- A brief explanation (1–2 sentences) justifying the score

Criteria:
1. Accuracy: Does the Filipino translation correctly convey the English source text’s meaning, intent, and details?
2. Completeness: Are all semantic elements of the English source text translated into Filipino without omissions or additions?
3. Cultural Appropriateness: Does the translation respect Filipino cultural norms, idioms, and sensitivities (e.g., use of "po" and "opo" for respect, regional expressions)?

Here are some additional guidelines...
1. Prefer natural, idiomatic translations that convey the intended meaning over stiff, literal translations that fail to capture the true sense. 

Here are examples of correct evaluation outputs:

Example 1:
SOURCE: The children laughed and played under the afternoon sun.
TRANSLATION: Ang mga bata ay nagtawanan at naglaro sa ilalim ng hapon na araw.
OUTPUT:
{{
  "per_criterion": {{
     "Accuracy": {{"score": 4, "explanation": "Conveys meaning accurately but 'hapon na araw' is slightly less idiomatic."}},
     "Completeness": {{"score": 5, "explanation": "All details from the source are preserved."}},
     "Cultural Appropriateness": {{"score": 4, "explanation": "Appropriate for Filipino readers, though 'hapon na araw' is less common."}}
  }},
  "overall_comment": "Accurate, fluent, and natural translation. Captures the tone and meaning well."
}}

Example 2:
SOURCE: This would be easier if blood came in more colours.
TRANSLATION: Ito ay magiging mas madali kung ang dugo ay dumating sa mas maraming kulay.
OUTPUT:
{{
  "per_criterion": {{
     "Accuracy": {{"score": 2, "explanation": "Literal translation misses the idiomatic meaning of 'came in more colors'."}},
     "Completeness": {{"score": 2, "explanation": "All elements are translated, but meaning is distorted."}},
     "Cultural Appropriateness": {{"score": 2, "explanation": "No cultural adaptation to convey intended tone or humor."}}
  }},
  "overall_comment": "Flawed translation takes 'came in more colors' literally."
}}

Example 3:
SOURCE: Adobo and sinigang are just some of the delicious Filipino dishes that are truly craved.
TRANSLATION: Ang adobo at sinigang ay ilan lamang sa masasarap na pagkaing Pilipino na talaga naman hinahanap-hanap
OUTPUT:
{{
  "per_criterion": {{
     "Accuracy": {{"score": 5, "explanation": "Perfectly conveys meaning and intent."}},
     "Completeness": {{"score": 5, "explanation": "All elements included without omissions or additions."}},
     "Cultural Appropriateness": {{"score": 5, "explanation": "Uses natural Filipino expression with proper repetition style."}}
  }},
  "overall_comment": "Sounds natural, and the repetition with '-' in 'hinahanap-hanap' is used correctly."
}}

Example 4:
SOURCE: The wave returns to the ocean, where it came from, and where it's supposed to be.
TRANSLATION: Bumabalik ang alon sa karagatan, kung saan ito nanggaling, at kung saan ito dapat nasa.
OUTPUT:
{{
  "per_criterion": {{
     "Accuracy": {{"score": 2, "explanation": "Meaning is partially preserved, but 'dapat nasa' is ungrammatical."}},
     "Completeness": {{"score": 3, "explanation": "Covers most of the meaning but loses nuance."}},
     "Cultural Appropriateness": {{"score": 3, "explanation": "No cultural issues, but language feels forced."}}
  }},
  "overall_comment": "The wording feels awkward and unnatural."
}}

Now, evaluate the following translation:

SOURCE:
{source}

TRANSLATION:
{translation}

Return ONLY the JSON in the same format as above.
"""

PROMPT_TEMPLATE_FLUENCY = """
You are a judge specialized in Filipino (Tagalog) sentence evaluation.

Given:
- A Filipino Sentence

Evaluate the FLUENCY on a score from 1 (worst) to 5 (best) using the following criteria:
Fluency: Is the sentence grammatically correct, natural, and idiomatic in Filipino?

Additional guidelines:
1. Penalize unnatural “ay” inversion in fluency scoring (e.g., ako ay kumain ng pansit) when it sounds forced or overly formal. But don't penalize it if it sounds natural in context.
2. Ensure correct use of direct nouns (nouns following "ang"), verifying proper syntax and meaning. Examples of proper use include:
    - Naghugas si Tatay ng pinggan sa lababo (Tatay washed dishes in the sink)
    - Hinugasan ni Tatay ang pinggan sa lababo (Tatay washed the dishes in the sink)
    - Kinainan ni Tatay ng chicken ang restawran (Tatay ate chicken at the restaurant)

Here are examples of correct evaluation outputs:

Example 1:
SENTENCE: Ang mga bata ay nagtawanan at naglaro sa ilalim ng hapon na araw.
OUTPUT:
{{
  "per_criterion": {{
     "Fluency": {{"score": 4, "explanation": "Natural structure but minor awkwardness in phrasing."}}
  }}
}}

Example 2:
SENTENCE: Ito ay magiging mas madali kung ang dugo ay dumating sa mas maraming kulay.
OUTPUT:
{{
  "per_criterion": {{
     "Fluency": {{"score": 2, "explanation": "Grammatically okay but sounds stiff and unnatural."}}
  }}
}}

Example 3:
SENTENCE: Ang adobo at sinigang ay ilan lamang sa masasarap na pagkaing Pilipino na talaga naman hinahanap-hanap
OUTPUT:
{{
  "per_criterion": {{
     "Fluency": {{"score": 5, "explanation": "Flows naturally with correct grammar and idiomatic use."}}
  }}
}}

Example 4:
SENTENCE: Bumabalik ang alon sa karagatan, kung saan ito nanggaling, at kung saan ito dapat nasa.
OUTPUT:
{{
  "per_criterion": {{
     "Fluency": {{"score": 1, "explanation": "Awkward and unnatural phrasing makes it difficult to read."}}
  }}
}}

Now, evaluate the following sentence:

SENTENCE:
{translation}

Return ONLY the JSON in the same format as above.
"""

rate_limit_num = 13
rate_limit_wait = 63
current_requests = 0

def call_gemini(prompt: str):
    global current_requests
    global rate_limit_num
    global rate_limit_wait

    current_requests += 1

    if current_requests >= rate_limit_num:
            print()
            print(f"Rate limit exceeded: {current_requests} > {rate_limit_num}")
            time.sleep(rate_limit_wait)
            current_requests = 0

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini call failed: {e}")

def parse_json_response(text: str):
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError("No JSON object found.")
    obj = json.loads(text[start:end+1])
    return obj


def evaluate_row(source_en: str, target_fil: str, retries=2, fluency=False):
    if fluency is False:
        prompt = PROMPT_TEMPLATE.format(source=source_en, translation=target_fil)
    else:
        prompt = PROMPT_TEMPLATE_FLUENCY.format(translation=target_fil)
    last_err = None
    for attempt in range(retries+1):
        try:
            raw = call_gemini(prompt)
            parsed = parse_json_response(raw)
            return {"raw": raw, "parsed": parsed}
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(f"Failed after retries: {last_err}")

def run_batch(dataset_path: str, out_path: str):
    df = pd.read_csv(dataset_path)
    required = {"Source Text (English)", "Target Text (Filipino)", "Final Score"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required}")

    results = []
    consistency_variations = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):

        try:
            # Normal evaluation
            res = evaluate_row(row["Source Text (English)"], row["Target Text (Filipino)"])
            global current_requests
            parsed = res["parsed"]

            entry = {
                "Source Text (English)": row["Source Text (English)"],
                "Target Text (Filipino)": row["Target Text (Filipino)"],
                "Overall_Comment": parsed.get("overall_comment", None)
            }

            predicted_score = 0
            total_weight = 0
            fluency_weight = 0
            for crit, weight in CRITERIA:
                if crit == "Fluency":
                    fluency_weight = weight
                    continue
                entry[f"{crit}_Score"] = parsed["per_criterion"][crit]["score"]
                entry[f"{crit}_Explanation"] = parsed["per_criterion"][crit]["explanation"]
                predicted_score += parsed["per_criterion"][crit]["score"] * weight
                total_weight += weight

            res_f = evaluate_row(row["Source Text (English)"], row["Target Text (Filipino)"], fluency=True)
            parsed_f = res_f["parsed"]

            entry["Fluency_Score"] = parsed_f["per_criterion"]["Fluency"]["score"]
            entry["Fluency_Explanation"] = parsed_f["per_criterion"]["Fluency"]["explanation"]
            predicted_score += parsed_f["per_criterion"]["Fluency"]["score"] * fluency_weight
            total_weight += fluency_weight

            entry["Predicted_Score"] = predicted_score / total_weight if total_weight > 0 else None
            entry["Validation Score"] = row["Final Score"]
            results.append(entry)

            print()
            print(f"Processed row {row.name}: ")
            print(f"  Source: {row['Source Text (English)']}")
            print(f"  Target: {row['Target Text (Filipino)']}")
            print(f"  Validation Score: {row['Final Score']}")
            print(f"  Predicted Score: {entry['Predicted_Score']:.2f}")
            print(f"    Accuracy: {entry['Accuracy_Score']}, Explanation: {entry['Accuracy_Explanation']}")
            print(f"    Fluency: {entry['Fluency_Score']}, Explanation: {entry['Fluency_Explanation']}")
            print(f"    Completeness: {entry['Completeness_Score']}, Explanation: {entry['Completeness_Explanation']}")
            print(f"    Cultural Appropriateness: {entry['Cultural Appropriateness_Score']}, Explanation: {entry['Cultural Appropriateness_Explanation']}")
            print(f"  Overall Comment: {entry['Overall_Comment']}")
            print()

            # Consistency check every 10th row
            if idx % 10 == 0:
                repeated_scores = []
                for _ in range(3):
                    res_c = evaluate_row(row["Source Text (English)"], row["Target Text (Filipino)"])
                    parsed_c = res_c["parsed"]

                    score_c = 0
                    total_w_c = 0
                    for crit, weight in CRITERIA:
                        if crit == "Fluency":
                            continue
                        score_c += parsed_c["per_criterion"][crit]["score"] * weight
                        total_w_c += weight

                    res_f_c = evaluate_row(row["Source Text (English)"], row["Target Text (Filipino)"], fluency=True)
                    parsed_f_c = res_f_c["parsed"]

                    fluency_score_c = parsed_f_c["per_criterion"]["Fluency"]["score"]
                    score_c += fluency_score_c * fluency_weight
                    total_w_c += fluency_weight

                    repeated_scores.append(score_c / total_w_c)

                variation = np.std(repeated_scores) / np.mean(repeated_scores) * 100
                consistency_variations.append({
                    "row_index": idx,
                    "english_sentence": row["Source Text (English)"],
                    "filipino_sentence": row["Target Text (Filipino)"],
                    "scores": repeated_scores,
                    "variation_percent": variation
                })
                print()
                print(f"Consistency test row {idx} → scores={repeated_scores} variation={variation:.2f}%")

        except Exception as e:
            print()
            print(f"Error processing row {row.name}:")
            traceback.print_exc()
            results.append({
                "Source Text (English)": row["Source Text (English)"],
                "Target Text (Filipino)": row["Target Text (Filipino)"],
                "Error": str(e)
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_path + "_results.csv", index=False)
    print(f"Saved evaluations to {out_path}")

    consistency_df = pd.DataFrame(consistency_variations)
    consistency_df.to_csv(out_path + "_consistency.csv", index=False)

    if consistency_variations:
        avg_var = np.mean([c["variation_percent"] for c in consistency_variations])
        max_var = np.max([c["variation_percent"] for c in consistency_variations])
        print(f"\nConsistency results across sampled rows:")
        print(f"  Avg variation: {avg_var:.2f}%")
        print(f"  Max variation: {max_var:.2f}%")
        print(f"Consistency Percentages: {[c['variation_percent'] for c in consistency_variations]}")

    # Spearman correlation
    if "Final Score" in df.columns:
        merged = out_df.merge(df[["Source Text (English)", "Final Score"]],
                              on="Source Text (English)", how="left")
        paired = merged.dropna(subset=["Predicted_Score", "Final Score"])
        if len(paired) >= 2:
            rho, pval = spearmanr(paired["Predicted_Score"], paired["Final Score"])
            print(f"Spearman ρ = {rho:.4f} (p={pval:.4g}) on {len(paired)} samples")

def run_single(english_text, filipino_text, out_path):

    res = evaluate_row(english_text, filipino_text)
    parsed = res["parsed"]

    entry = {
        "Source Text (English)": english_text,
        "Target Text (Filipino)": filipino_text,
        "LLM_JSON": json.dumps(parsed, ensure_ascii=False),
        "Overall_Comment": parsed.get("overall_comment", None)
    }

    predicted_score = 0
    total_weight = 0
    fluency_weight = 0
    for crit, weight in CRITERIA:
        if crit == "Fluency":
            fluency_weight = weight
            continue
        entry[f"{crit}_Score"] = parsed["per_criterion"][crit]["score"]
        entry[f"{crit}_Explanation"] = parsed["per_criterion"][crit]["explanation"]
        predicted_score += parsed["per_criterion"][crit]["score"] * weight
        total_weight += weight

    res_f = evaluate_row(english_text, filipino_text, fluency=True)
    parsed_f = res_f["parsed"]

    entry["Fluency_Score"] = parsed_f["per_criterion"]["Fluency"]["score"]
    entry["Fluency_Explanation"] = parsed_f["per_criterion"]["Fluency"]["explanation"]
    predicted_score += parsed_f["per_criterion"]["Fluency"]["score"] * fluency_weight
    total_weight += fluency_weight

    entry["Predicted_Score"] = predicted_score / total_weight if total_weight > 0 else None

    print()
    print(f"Source: {english_text}")
    print(f"Target: {filipino_text}")
    print(f"Predicted Score: {entry['Predicted_Score']:.2f}")
    print(f"  Accuracy: {entry['Accuracy_Score']}, Explanation: {entry['Accuracy_Explanation']}")
    print(f"  Fluency: {entry['Fluency_Score']}, Explanation: {entry['Fluency_Explanation']}")
    print(f"  Adequacy: {entry['Completeness_Score']}, Explanation: {entry['Completeness_Explanation']}")
    print(f"  Cultural Appropriateness: {entry['Cultural Appropriateness_Score']}, Explanation: {entry['Cultural Appropriateness_Explanation']}")
    print(f"Overall Comment: {entry['Overall_Comment']}")
    print()

    with open(out_path + ".json", "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path to dataset CSV")
    parser.add_argument("--out", required=True, help="Output File Name")
    parser.add_argument("--english", help="English sentence")
    parser.add_argument("--filipino", help="Filipino sentence")
    args = parser.parse_args()

    if args.english and args.filipino:
        run_single(args.english, args.filipino, args.out)
    else:
        run_batch(args.dataset, args.out)
