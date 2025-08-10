import os
import json
import time
import argparse

import numpy as np
import pandas as pd

import google.generativeai as genai

from agent_tools import get_english_definition
from agent_tools import get_tagalog_definition_2
from agent_tools import labse_similarity
from agent_tools import TranslationJudgeMemory

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your environment.")
genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"

current_requests = 0
rate_limit_num = 13
rate_limit_wait = 62

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
        print(text)
        raise ValueError("No JSON object found.")
    obj = json.loads(text[start:end+1])
    return obj

MAIN_PROMPT = """
You are the Orchestration Layer of a Filipino-English LLM Judge system.

Your job is to decide whether the given information (in JSON format) is enough to produce a final judgment on translation quality, and whether or not to ask the user for additional information.
{json}

You must follow these decision rules:
1. The fields "english_sentence" and "filipino_sentence" are absolutely essential. Do NOT proceed without them. If they are already present, do not re-ask for them.
2. Check if the system has already asked for any **additional information** needed for judging (in "user_notes" and the variable "has_asked_for_additional_info").
   - If YES, mark as complete.
   - If NO, add a step to ask the user for it.
   Do not force the user to provide it. If the user notes indicate that there are no more context or additional instructions, mark this step as complete. 
3. Check if the system has asked the user if the current weights are acceptable (in "user_notes" and the variable "has_asked_if_weights_are_acceptable").
   - If YES, mark as complete.  
   - If NO, add a step to ask the user for it, supplying the current weights in natural language. Make sure to indicate that the weights values are only among "High", "Medium", and "Low".
   Do not force the user to provide it. If the user notes indicate that the existing weights are alright, mark this step as complete.

**Your output must be ONLY a JSON object in the following format:**
{{
  "reason": "Reasoning for the steps below.",
  "steps": [
    "ask \\\"Please provide the English sentence and the Filipino translation.\\\"",
    "ask \\\"Please provide additional context or instructions for the translation.\\\""
  ]
}}

Where:
- `"steps"` is the ordered list of actions the system must take next if ready_to_decide is FALSE. 
Do NOT add any ask steps that have already been asked in the past_questions list. 

As long as it's reasonable, combine all ask steps into one.

Return ONLY valid JSON. Make sure to escape all double quotes. No extra text.
"""

TOOL_PROMPT = """
You are the Orchestration Layer of a Filipino-English LLM Judge system.

Your job is to decide whether the given information (in JSON format) is enough to produce a final judgment on translation quality, and whether or not to use certain tools to supplement the information.
{json}

You must follow these decision rules:
1. Determine if the system needs to use any tools to proceed. If YES, add those tool steps.  
   If NO, then do not add any tool steps.

Available tools and their command formats:
- **Sentence similarity**: `similarity "english phrase/sentence" "filipino phrase/sentence"`
- **English Dictionary lookup**: `lookup_english "singular word"`
- **Filipino Dictionary lookup**: `lookup_filipino "singular word"`
Use them as often as you need, preferably for difficult cases, but do not use them too much or excessively.
Only add tool steps that are strictly necessary to clarify difficult or ambiguous words or sentence similarity.
For dictionary lookups, STRICTLY ONLY SINGULAR WORDS, NO PHRASES.
And no repeat queries; consult with the banned words list below STRICTLY.

**Your output must be ONLY a JSON object in the following format:**
{{
  "reason": "Reasoning for the steps below.",
  "steps": [
    "similarity \\\"The cat is on the roof.\\\" \\\"Ang pusa ay nasa bubong.\\\"",
    "lookup_english \\\"cat\\\"",
    "lookup_filipino \\\"pusa\\\""
  ]
}}

Where:
- `"steps"` is the ordered list of actions the system must take next if ready_to_decide is FALSE. 

{filipino_words_queried}
{english_words_queried}
{similarities_queried}

Return ONLY valid JSON. Make sure to escape all double quotes. No extra text.
"""

UPDATE_PROMPT = """
You are the Orchestration Layer of a Filipino-English LLM Judge system.

Your job is to take in the user's input and extract the variables to be added to memory (in JSON format). The current memory is this:
{json}

Follow this format for adding to the memory and output lines based on what to add:
1. If you see that the user has supplied the English or Filipino text, then add:
english_sentence "(English sentence)"
filipino_sentence "(Filipino sentence)"
2. If the user has provided additional guidelines/instructions or has stated that there are no additional guidelines/instructions, then add it in the user notes and mark it as completed:
note "(the additional instructions of the sentences)"
has_asked_for_additional_instructions True
3. If the user has stated that no context or no additional instructions are to be provided, then add it in the user notes. Make sure that note has context of the question.
4. If the user has indicated that they want to change the weights of a particular criterion, then add:
    accuracy_weight "(new weight)"
    fluency_weight "(new weight)"
    completeness_weight "(new weight)"
    cultural_appropriateness_weight "(new weight)"
    has_asked_if_weights_are_acceptable True
Make sure that the new weights are only among "High", "Medium", and "Low".

You must output ONLY valid JSON.
No extra text, code blocks, or explanations.
Do not add extra quotation marks.
Make sure to escape all double quotes.

Your output must be ONLY a JSON object in the following format:**
{{
  "updates": [
    "english_sentence \\\"The cat is on the roof.\\\"",
    "filipino_sentence \\\"Ang pusa ay nasa bubong.\\\"",
    accuracy_weight \\\"High\\\"
    fluency_weight \\\"High\\\"
    completeness_weight \\\"High\\\"
    cultural_appropriateness_weight \\\"High\\\"
    "note \\\"There is no more additional information.\\\""
    has_asked_for_additional_instructions True
    has_asked_if_weights_are_acceptable True
  ]
}}

The user was asked the following question:
{question}

The user has provided the following information:
{input}

Return ONLY valid JSON. No extra text.
"""

FINAL_JUDGE_PROMPT = """
You are the final expert judge in a Filipino-English LLM Judge system.

Your job is to provide the final verdict on the score of a sentence pair based on specific criteria.

Given:
- SOURCE (English)
- TRANSLATION (Filipino)
- Additional notes from the user
- Additional notes from earlier tool usage (dictionary lookups and similarity lookups)

Evaluate the TRANSLATION using the following criteria. For each criterion, give:
- A score from 1 (worst) to 5 (best)
- A brief explanation (1–2 sentences) justifying the score

Criteria:
1. Accuracy: Does the Filipino translation correctly convey the English source text’s meaning, intent, and details?
2. Completeness: Are all semantic elements of the English source text translated into Filipino without omissions or additions?
3. Cultural Appropriateness: Does the translation respect Filipino cultural norms, idioms, and sensitivities (e.g., use of "po" and "opo" for respect, regional expressions)?

Additional guidelines:
- Prefer natural, idiomatic translations that convey the intended meaning over stiff, literal translations that fail to capture the true sense.
- You can be harsh with your scoring if the translation is inaccurate or fails to meet the criteria.

Examples of correct evaluation outputs:

Example 1:
SOURCE: The children laughed and played under the afternoon sun.
TRANSLATION: Ang mga bata ay nagtawanan at naglaro sa ilalim ng hapon na araw.
USER NOTES:
- The context is from a movie that depicts childhood innocence and joy.
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
TOOL NOTES:
- Top results for Tagalog word 'madali': 1: madali: [adjective]easy • not difficult • straight forward • convenient 
OUTPUT:
{{
  "per_criterion": {{
     "Accuracy": {{"score": 2, "explanation": "Literal translation misses the idiomatic meaning of 'came in more colors'."}},
     "Completeness": {{"score": 2, "explanation": "All elements are translated, but meaning is distorted."}},
     "Cultural Appropriateness": {{"score": 2, "explanation": "No cultural adaptation to convey intended tone or humor."}}
  }},
  "overall_comment": "Flawed translation takes 'came in more colors' literally."
}}

Now, evaluate the following translation:

SOURCE:
{source}

TRANSLATION:
{translation}

{user_notes}{tool_notes}

Return ONLY the JSON in the same format as above.
"""

FINAL_JUDGE_PROMPT_FLUENCY = """
You are a judge specialized in Filipino (Tagalog) sentence evaluation.

Given:
- A Filipino Sentence
- Additional notes from the user
- Additional notes from earlier tool usage (dictionary lookups and similarity lookups)

Evaluate the FLUENCY on a score from 1 (worst) to 5 (best) using the following criteria:
Fluency: Is the sentence grammatically correct, natural, and idiomatic in Filipino?

Additional guidelines:
1. Penalize unnatural “ay” inversion in fluency scoring (e.g., ako ay kumain ng pansit) when it sounds forced or overly formal. But don't penalize it if it sounds natural in context.
2. Ensure correct use of direct nouns (nouns following "ang"), verifying proper syntax and meaning. Examples of proper use include:
    - Naghugas si Tatay ng pinggan sa lababo (Tatay washed dishes in the sink)
    - Hinugasan ni Tatay ang pinggan sa lababo (Tatay washed the dishes in the sink)
    - Kinainan ni Tatay ng chicken ang restawran (Tatay ate chicken at the restaurant)
3. You can be harsh with your scoring if the translation is inaccurate or fails to meet the criteria.


Examples of correct evaluation outputs:

Example 1:
SOURCE: The children laughed and played under the afternoon sun.
TRANSLATION: Ang mga bata ay nagtawanan at naglaro sa ilalim ng hapon na araw.
USER NOTES:
- The context is from a movie that depicts childhood innocence and joy.
OUTPUT:
{{
  "per_criterion": {{
     "Fluency": {{"score": 4, "explanation": "Natural structure but minor awkwardness in phrasing."}}
  }}
}}

Example 2:
SOURCE: This would be easier if blood came in more colours.
TRANSLATION: Ito ay magiging mas madali kung ang dugo ay dumating sa mas maraming kulay.
TOOL NOTES:
- Top results for Tagalog word 'madali': 1: madali: [adjective]easy • not difficult • straight forward • convenient 
OUTPUT:
{{
  "per_criterion": {{
     "Fluency": {{"score": 2, "explanation": "Grammatically okay but sounds stiff and unnatural."}}
  }}
}}

Now, evaluate the following sentence:

SENTENCE:
{source}

{user_notes}{tool_notes}

Return ONLY the JSON in the same format as above.
"""

def evaluate_row(memory, retries=2, fluency=False):
    # Prepare user_notes section string
    if memory.memory.get("user_notes"):
        user_notes_section = "USER NOTES:\n" + "\n".join(memory.memory.get("user_notes")) + "\n"
    else:
        user_notes_section = ""

    # Prepare tool_notes section string
    if memory.memory.get("tool_notes"):
        tool_notes_section = "TOOL NOTES:\n" + "\n".join(memory.memory.get("tool_notes")) + "\n"
    else:
        tool_notes_section = ""

    if fluency is False:
        prompt = FINAL_JUDGE_PROMPT.format(source=memory.memory["english_sentence"], translation=memory.memory["filipino_sentence"], user_notes=user_notes_section, tool_notes=tool_notes_section)
    else:
        prompt = FINAL_JUDGE_PROMPT_FLUENCY.format(source=memory.memory["filipino_sentence"], user_notes=user_notes_section, tool_notes=tool_notes_section)

    # print(prompt)
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

def main_loop(english_sentence=None, filipino_sentence=None, user_notes=[], has_asked_for_additional_instructions=False, has_asked_if_weights_are_acceptable=False):
    past_thoughts = []

    memory = TranslationJudgeMemory()
    memory.update("english_sentence", english_sentence)
    memory.update("filipino_sentence", filipino_sentence)
    memory.update("has_asked_for_additional_instructions", has_asked_for_additional_instructions)
    memory.update("has_asked_if_weights_are_acceptable", has_asked_if_weights_are_acceptable)

    for note in user_notes:
        memory.add_note(note, source="user")

    limit = 5
    iters = 0

    while True:

        steps = []
        if memory.memory["english_sentence"] is None or memory.memory["filipino_sentence"] is None or iters < limit:
            iters += 1

            json1_str = memory.__str__()
            json1 = json.loads(json1_str)

            del json1["tool_notes"]
            del json1["english_words_queried"]
            del json1["filipino_words_queried"]
            del json1["similarities_queried"]
            json1_clean_str = json.dumps(json1, indent=4)

            main_prompt = MAIN_PROMPT.format(json=json1_clean_str)
            #print("\n---\n")
            #print(main_prompt)
            res = call_gemini(main_prompt)
            res = parse_json_response(res)
            steps.extend(res.get("steps", []))
            past_thoughts.append(f"Reason: {res['reason']}")
            past_thoughts.append(f"Next steps: {res['steps']}")
        # print()
            #print(res)
            #print("\n---\n")

            if memory.memory["english_sentence"] is not None and memory.memory["filipino_sentence"] is not None:

                json_str_2 = memory.__str__()
                json_dict_2 = json.loads(json_str_2)

                for key in [
                    "accuracy_weight",
                    "fluency_weight",
                    "completeness_weight",
                    "cultural_appropriateness_weight",
                    "user_notes",
                    "past_questions",
                    "has_asked_for_additional_instructions",
                    "has_asked_if_weights_are_acceptable",
                ]:
                    json_dict_2.pop(key, None)  

                english_words_queried = json_dict_2.get("english_words_queried", [])
                filipino_words_queried = json_dict_2.get("filipino_words_queried", [])
                similarities_queried = json_dict_2.get("similarities_queried", [])

                english_words_queried_str = ", ".join(english_words_queried)
                filipino_words_queried_str = ", ".join(filipino_words_queried)
                similarities_queried_str = ", ".join(similarities_queried)

                if english_words_queried:
                    english_words_queried_str = "UNDER NO CIRCUMSTANCES should you query these English words since they were already queried: " + ", ".join(english_words_queried)

                if filipino_words_queried:
                    filipino_words_queried_str = "UNDER NO CIRCUMSTANCES should you query these Filipino words since they were already queried: " + ", ".join(filipino_words_queried)

                if similarities_queried:
                    similarities_queried_str = "UNDER NO CIRCUMSTANCES should you query these similarities since they were already queried: " + ", ".join(similarities_queried)

                for key in [
                    "english_words_queried",
                    "filipino_words_queried",
                    "similarities_queried"
                ]:
                    json_dict_2.pop(key, None) 

                # If you want JSON string back
                clean_json_str_2 = json.dumps(json_dict_2, indent=4)

                tool_prompt = TOOL_PROMPT.format(json=clean_json_str_2, english_words_queried=english_words_queried_str, filipino_words_queried=filipino_words_queried_str, similarities_queried=similarities_queried_str)
                #print("\n-======-\n")
                #print(tool_prompt)
                res2 = call_gemini(tool_prompt)
                res2 = parse_json_response(res2)
                steps.extend(res2.get("steps", []))
                past_thoughts.append(f"Reason: {res2['reason']}")
                past_thoughts.append(f"Next steps: {res2['steps']}")
                #print()
                #print(res2)
                #print("\n-======-\n")

        if not steps:
            res = evaluate_row(memory, retries=2, fluency=False)
            parsed = res["parsed"]

            # print("----")
            # print(json.dumps(parsed, indent=4, ensure_ascii=False))

            entry = {
                "Source Text (English)": memory.memory["english_sentence"],
                "Target Text (Filipino)": memory.memory["filipino_sentence"],
                "Overall_Comment": parsed.get("overall_comment", None)
            }

            predicted_score = 0
            total_weight = 0
            fluency_weight = 0
            for crit, crit2 in [("Accuracy", "accuracy"), ("Fluency", "fluency"), ("Completeness", "completeness"), ("Cultural Appropriateness", "cultural_appropriateness")]:
                weight = memory.memory.get(f"{crit2}_weight", "High")

                if weight == "High":
                    weight = 1.5
                elif weight == "Medium":
                    weight = 1.0
                elif weight == "Low":
                    weight = 0.5

                if crit == "Fluency":
                    fluency_weight = weight
                    continue
                entry[f"{crit}_Score"] = parsed["per_criterion"][crit]["score"]
                entry[f"{crit}_Explanation"] = parsed["per_criterion"][crit]["explanation"]
                predicted_score += parsed["per_criterion"][crit]["score"] * weight
                total_weight += weight

            res_f = evaluate_row(memory, retries=2, fluency=True)
            parsed_f = res_f["parsed"]

            entry["Fluency_Score"] = parsed_f["per_criterion"]["Fluency"]["score"]
            entry["Fluency_Explanation"] = parsed_f["per_criterion"]["Fluency"]["explanation"]
            predicted_score += parsed_f["per_criterion"]["Fluency"]["score"] * fluency_weight
            total_weight += fluency_weight

            entry["Predicted_Score"] = predicted_score / total_weight if total_weight > 0 else None

            # print(json.dumps(entry, indent=4, ensure_ascii=False))
            return entry, memory, past_thoughts

        else:

            for step in steps:
               #print(memory)
                #print()
                if step.startswith("ask"):
                    question = step.split("\"")[1]
                    print()
                    print(question)
                    memory.add_past_question(question)

                    user_input = input("Input: ")
                    update_prompt = UPDATE_PROMPT.format(json=memory.__str__(), input=user_input, question=question)
                    res2 = call_gemini(update_prompt)
                    # print(res2)
                    res2 = parse_json_response(res2)
                    for update in res2['updates']:
                        if update.startswith("english_sentence"):
                            memory.update("english_sentence", update.split("\"")[1])
                        elif update.startswith("filipino_sentence"):
                            memory.update("filipino_sentence", update.split("\"")[1])
                        elif update.startswith("accuracy_weight"):
                            new_weight = update.split("\"")[1]
                            memory.update("accuracy_weight", new_weight)
                           #  memory.add_note(f"Updated accuracy weight to {new_weight}", source="user")
                        elif update.startswith("fluency_weight"):
                            new_weight = update.split("\"")[1]
                            memory.update("fluency_weight", new_weight)
                            # memory.add_note(f"Updated fluency weight to {new_weight}", source="user")
                        elif update.startswith("completeness_weight"):
                            new_weight = update.split("\"")[1]
                            memory.update("completeness_weight", new_weight)
                            # memory.add_note(f"Updated completeness weight to {new_weight}", source="user")
                        elif update.startswith("cultural_appropriateness_weight"):
                            new_weight = update.split("\"")[1]
                            memory.update("cultural_appropriateness_weight", new_weight)
                            # memory.add_note(f"Updated cultural appropriateness weight to {new_weight}", source="user")
                        elif update.startswith("note"):
                            memory.add_note(update.split("\"")[1], source="user")
                        elif update.startswith("has_asked_for_additional_instructions"):
                            memory.update("has_asked_for_additional_instructions", update.split(" ")[1] == "True")
                        elif update.startswith("has_asked_if_weights_are_acceptable"):
                            memory.update("has_asked_if_weights_are_acceptable", update.split(" ")[1] == "True")

                elif step.startswith("similarity"):
                    phrases = step.split("\"")[1::2]
                    similarity = labse_similarity(phrases[0], phrases[1])

                    if phrases[0] + " | " + phrases[1] + " | " + str(similarity) in memory.memory.get("similarities_queried", []):
                        continue

                    print()
                    print("Checking similarity between:")
                    print(f" - {phrases[0]}")
                    print(f" - {phrases[1]}")
                    print()
                    
                    memory.add_note(f"Similarity between '{phrases[0]}' and '{phrases[1]}': {similarity}", source="tool")
                    memory.add_similarity(phrases[0] + " | " + phrases[1] + " | " + str(similarity))
                elif step.startswith("lookup_english"):
                    word = step.split("\"")[1]
                    if word in memory.memory.get("english_words_queried", []):
                        continue

                    print(f"Looking up English word: {word}")
                    print()
                    definition = get_english_definition(word)
                    if definition:
                        memory.add_note(f"English definition of '{word}': {definition}", source="tool")
                        memory.add_english_word(word)
                    else:
                        memory.add_note(f"Could not find English definition for '{word}'", source="tool")
                        memory.add_english_word(word)
                elif step.startswith("lookup_filipino"):
                    word = step.split("\"")[1]
                    if word in memory.memory.get("filipino_words_queried", []):
                        continue

                    print(f"Looking up Filipino word: {word}")
                    print()
                    results = get_tagalog_definition_2(word)
                    # print(results)
                    if results:
                        note = f"Top results for Tagalog word '{word}': "
                        for idx, res in enumerate(results, 1):
                            temp = f"{idx}: {res['search_result']}: {res['definition']} "
                            temp = temp.replace(" Example Sentences", "")
                            note += temp
                        memory.add_note(note, source="tool")
                        memory.add_filipino_word(word)
                    else:
                        memory.add_note(f"Could not find Filipino definition for '{word}'", source="tool")
                        memory.add_filipino_word(word)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path to dataset CSV")
    parser.add_argument("--out", required=True, help="Output File Name")
    parser.add_argument("--consistency", action="store_true", help="Consistency Check")
    args = parser.parse_args()
    
    if args.dataset and args.consistency:
        df = pd.read_csv(args.dataset)
        required = {"Source Text (English)", "Target Text (Filipino)", "Final Score"}
        if not required.issubset(df.columns):
            raise ValueError(f"Dataset must contain columns: {required}")
        try:
            consistency_variations = [] 
            for idx, row in df.iterrows():
                if idx % 10 != 0:
                    continue

                repeated_scores = []
                for _ in range(3):
                    english_sentence = row["Source Text (English)"]
                    filipino_sentence = row["Target Text (Filipino)"]
                    user_notes = ["No more additional context, instructions, or information, available. Do not ask anymore.", 'The weights are fine as is. Do not ask anymore.']
                    result, _, _ = main_loop(english_sentence, filipino_sentence, user_notes, True, True)
                    repeated_scores.append(result["Predicted_Score"])
                
                variation = np.std(repeated_scores) / np.mean(repeated_scores) * 100
                consistency_variations.append({
                    "row_index": idx,
                    "english_sentence": english_sentence,
                    "filipino_sentence": filipino_sentence,
                    "scores": repeated_scores,
                    "variation_percent": variation
                })
                print()
                print(f"Consistency test row {idx} → scores={repeated_scores} variation={variation:.2f}%")
            avg_var = np.mean([c["variation_percent"] for c in consistency_variations])
            max_var = np.max([c["variation_percent"] for c in consistency_variations])
            print(f"\nConsistency results across sampled rows:")
            print(f"  Avg variation: {avg_var:.2f}%")
            print(f"  Max variation: {max_var:.2f}%")
            print(f"Consistency Percentages: {[c['variation_percent'] for c in consistency_variations]}")
        except Exception as e:
            print(e)
            print("Error processing dataset. Exiting.")
        finally:
            output_df = pd.DataFrame(consistency_variations)
            output_df.to_csv(args.out + "_consistency.csv", index=False, encoding='utf-8-sig')


    elif args.dataset:
        df = pd.read_csv(args.dataset)
        required = {"Source Text (English)", "Target Text (Filipino)", "Final Score"}
        if not required.issubset(df.columns):
            raise ValueError(f"Dataset must contain columns: {required}")
        
        results = []
        try:
            for _, row in df.iterrows():
                english_sentence = row["Source Text (English)"]
                filipino_sentence = row["Target Text (Filipino)"]
                user_notes = ["No more additional context, instructions, or information, available. Do not ask anymore.", 'The weights are fine as is. Do not ask anymore.']
                result, _, _ = main_loop(english_sentence, filipino_sentence, user_notes, True, True)
                result["Validation Score"] = row["Final Score"]
                results.append(result)
                print(result)
        except Exception as e:
            print(e)
            print("Error processing dataset. Exiting.")
        finally:
            output_df = pd.DataFrame(results)
            output_df.to_csv(args.out + "_results.csv", index=False, encoding='utf-8-sig')
    else:
        result, memory, thoughts = main_loop()
        print(memory)
        print(thoughts)
        print(result)

        # output all to json
        with open(args.out + "_result.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        with open(args.out + "_memory.json", 'w', encoding='utf-8') as f:
            json.dump(memory.memory, f, ensure_ascii=False, indent=4)
        with open(args.out + "_thoughts.json", 'w', encoding='utf-8') as f:
            json.dump(thoughts, f, ensure_ascii=False, indent=4)
