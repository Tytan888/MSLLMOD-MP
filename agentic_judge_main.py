import os
import json
import time
import argparse
import pandas as pd
import statistics
from scipy.stats import spearmanr
from tqdm import tqdm
import google.generativeai as genai
import numpy as np
import traceback
import requests
from transformers import AutoTokenizer, AutoModel
import torch
from huggingface_hub import hf_hub_download
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
from selenium.webdriver.chrome.service import Service

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your environment.")
genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.5-flash"

current_requests = 0
rate_limit_num = 10
rate_limit_wait = 62

def call_gemini(prompt: str):
    global current_requests
    global rate_limit_num
    global rate_limit_wait
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

def get_english_definition(word):
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        meanings = []
        for meaning in data[0]["meanings"]:
            for definition in meaning["definitions"]:
                meanings.append(definition["definition"])
        return meanings
    else:
        return None


def get_tagalog_definition(word):
    # Configure headless Chrome
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")
    options.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])
    
    service = Service(log_path=os.devnull)

    driver = webdriver.Chrome(options=options)

    try:
        url = f"https://www.tagalog.com/dictionary/{word}"
        driver.get(url)

        # Give time for Cloudflare & JS rendering
        time.sleep(2)

        soup = BeautifulSoup(driver.page_source, "html.parser")

        # First div with class "definition"
        first_def_div = soup.find("div", class_="definition")
        if not first_def_div:
            return None

        # Third div inside it
        inner_divs = first_def_div.find_all("div")
        if len(inner_divs) < 3:
            return None

        target_div = inner_divs[2]  # zero-indexed

        # Remove <span> tags
        for span in target_div.find_all("span"):
            span.unwrap()

        return target_div.get_text(strip=True)

    finally:
        driver.quit()

def get_tagalog_definition_2(word):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--enable-unsafe-swiftshader")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")

    # Suppress logging:
    options.add_argument("--log-level=3")  # Only fatal errors
    options.add_experimental_option('excludeSwitches', ['enable-logging'])  # suppress console logs

    driver = webdriver.Chrome(options=options)

    try:
        url = f"https://www.tagalog.com/#{word}"
        driver.get(url)

        time.sleep(2)  # wait for JS & Cloudflare

        soup = BeautifulSoup(driver.page_source, "html.parser")

        results = []

        for i in range(1, 4):
            result_id = f"search_result{i}"
            result_elem = soup.find(id=result_id)
            if not result_elem:
                continue

            # Clean search result inner HTML (unwrap spans)
            for span in result_elem.find_all("span"):
                span.unwrap()
            search_result_text = result_elem.get_text(strip=True)

            # First sibling div with definitions
            def_div = result_elem.find_next_sibling("div")
            if def_div:
                for span in def_div.find_all("span"):
                    span.unwrap()
                def_text = def_div.get_text(strip=True)
            else:
                def_text = None

            results.append({
                "search_result": search_result_text,
                "definition": def_text
            })

        return results

    finally:
        driver.quit()

# Load LaBSE model
similarity_model_name = "sentence-transformers/LaBSE"
similarity_tokenizer = AutoTokenizer.from_pretrained(similarity_model_name)
similarity_model = AutoModel.from_pretrained(similarity_model_name)

def labse_similarity(text1: str, text2: str) -> float:
    # Tokenize
    inputs = similarity_tokenizer([text1, text2], return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():
        embeddings = similarity_model(**inputs).pooler_output  # shape: (2, hidden_size)

    # Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # Cosine similarity
    similarity = torch.matmul(embeddings[0], embeddings[1]).item()
    return similarity

class TranslationJudgeMemory:
    def __init__(self):
        self.memory = {
            "english_sentence": None,
            "filipino_sentence": None,
            "accuracy_weight": 1.5,
            "fluency_weight": 1.5,
            "completeness_weight": 0.5,
            "cultural_appropriateness_weight": 0.5,
            "user_notes": [],
            "tool_notes": [],
            "past_questions": []
        }

    def update(self, key, value):
        """Update a memory field."""
        if key in self.memory:
            self.memory[key] = value
        else:
            raise KeyError(f"Memory key '{key}' not found.")

    def add_note(self, note, source="user"):
        """Add note to either user_notes or tool_notes."""
        if source == "user":
            self.memory["user_notes"].append(note)
        elif source == "tool":
            self.memory["tool_notes"].append(note)

    def add_past_question(self, question):
        """Add a past question to the memory."""
        self.memory["past_questions"].append(question)

    def __str__(self):
        return json.dumps(self.memory, ensure_ascii=False, indent=2)

MAIN_PROMPT = """
You are the Orchestration Layer of a Filipino-English LLM Judge system.

Your job is to decide whether the given information (in JSON format) is enough to produce a final judgment on translation quality.
{json}

You must follow these decision rules:
1. The fields "english_sentence" and "filipino_sentence" are absolutely essential. Do NOT proceed without them. If they are already present, do not re-ask for them.
2. Check if the system has already asked for the **context of the sentence** (in "user_notes").
   - If YES, mark as complete.  
   - If NO, add a step to ask the user for it.
   Do not force the user to provide it. If the user notes indicate that there is no context, mark this step as complete.
3. Check if the system has already asked for any **additional information** needed for judging (in "user_notes").  
   - If YES, mark as complete.  
   - If NO, add a step to ask the user for it.
   Do not force the user to provide it. If the user notes indicate that there are no more context or additional instructions, mark this step as complete.
4. Determine if the system needs to use any tools to proceed. If YES, add those tool steps.  
   If NO, list all tools that could still be used and their intended queries.
5. DO NOT repeat the same question twice. Refer to the past_questions list in the json memory.

Available tools and their command formats:
- **Sentence similarity**: `similarity "english phrase/sentence" "filipino phrase/sentence"`
- **English Dictionary lookup**: `lookup_english "word"`
- **Filipino Dictionary lookup**: `lookup_filipino "word"`
Use them often, preferably for difficult cases, but not excessively. But check the tool notes in the json above first to avoid repeating the same queries.

You must output ONLY valid JSON.
No extra text, code blocks, or explanations.
Do not add extra quotation marks.
Make sure to escape all double quotes.

**Your output must be ONLY a JSON object in the following format:**
{{
  "ready_to_decide": true/false,
  "reason": "Why you can or cannot decide yet",
  "steps": [
    "ask \\\"Please provide the English sentence and the Filipino translation.\\\"",
    "similarity \\\"The cat is on the roof.\\\" \\\"Ang pusa ay nasa bubong.\\\"",
    "lookup_english \\\"cat\\\"",
    "lookup_filipino \\\"pusa\\\""
  ]
}}

Where:
- `"ready_to_decide"` is TRUE if all required info is present, otherwise FALSE.
- `"steps"` is the ordered list of actions the system must take next if ready_to_decide is FALSE.

As long as it's reasonable, combine all ask steps into one.

Return ONLY valid JSON. No extra text.
"""

UPDATE_PROMPT = """
You are the Orchestration Layer of a Filipino-English LLM Judge system.

Your job is to take in the user's input and extract the variables to be added to memory (in JSON format). The current memory is this:
{json}

Follow this format for adding to the memory and output lines based on what to add:
1. If you see that the user has supplied the English or Filipino text, then add:
english_sentence "(English sentence)"
filipino_sentence "(Filipino sentence)"
2. If the user has provided the context of the sentence (i.e., what is its source or domain), add it in the user notes:
note "(the context of the sentences)"
3. If the user has provided additional guidelines/instructions, then add it in the user notes:
note "(the additional instructions of the sentences)"
4. If the user has stated that no context or no additional instructions are to be provided, then add it in the user notes.
5. DO NOT repeat past questions or similar ones. Refer to the past_questions list in the json memory.

You must output ONLY valid JSON.
No extra text, code blocks, or explanations.
Do not add extra quotation marks.
Make sure to escape all double quotes.

Your output must be ONLY a JSON object in the following format:**
{{
  "updates": [
    "english_sentence \\\"The cat is on the roof.\\\"",
    "filipino_sentence \\\"Ang pusa ay nasa bubong.\\\"",
    "note \\\"There is no context to this sentence. Just a simple statement about a cat on a roof.\\\""
  ]
}}

The user has provided the following information:
{input}

Return ONLY valid JSON. No extra text.
"""

def main_loop(english_sentence=None, filipino_sentence=None, user_notes=[]):
    past_thoughts = []

    memory = TranslationJudgeMemory()
    memory.update("english_sentence", english_sentence)
    memory.update("filipino_sentence", filipino_sentence)
    for note in user_notes:
        memory.add_note(note, source="user")

    while True:
        main_prompt = MAIN_PROMPT.format(json=memory.__str__())
        print(main_prompt)
        res = call_gemini(main_prompt)
        res = parse_json_response(res)
        print(res)

        if res["ready_to_decide"]:
            print("Ready to decide!")
            print(memory)
            print(past_thoughts)
            break
        else:
            past_thoughts.append(f"Reason: {res['reason']}")
            past_thoughts.append(f"Next steps: {res['steps']}")

            for step in res['steps']:
                print(memory)
                print()
                if step.startswith("ask"):
                    question = step.split("\"")[1]
                    print(question)
                    memory.add_past_question(question)

                    user_input = input("Input: ")
                    update_prompt = UPDATE_PROMPT.format(json=memory.__str__(), input=user_input)
                    res2 = call_gemini(update_prompt)
                    print(res2)
                    res2 = parse_json_response(res2)
                    for update in res2['updates']:
                        if update.startswith("english_sentence"):
                            memory.update("english_sentence", update.split("\"")[1])
                        elif update.startswith("filipino_sentence"):
                            memory.update("filipino_sentence", update.split("\"")[1])
                        elif update.startswith("note"):
                            memory.add_note(update.split("\"")[1], source="user")
                elif step.startswith("similarity"):
                    phrases = step.split("\"")[1::2]
                    print("Checking similarity between:")
                    print(f" - {phrases[0]}")
                    print(f" - {phrases[1]}")
                    print()
                    similarity = labse_similarity(phrases[0], phrases[1])
                    memory.add_note(f"Similarity between '{phrases[0]}' and '{phrases[1]}': {similarity}", source="tool")
                elif step.startswith("lookup_english"):
                    word = step.split("\"")[1]
                    print(f"Looking up English word: {word}")
                    print()
                    definition = get_english_definition(word)
                    if definition:
                        memory.add_note(f"English definition of '{word}': {definition}", source="tool")
                    else:
                        memory.add_note(f"Could not find English definition for '{word}'", source="tool")
                elif step.startswith("lookup_filipino"):
                    word = step.split("\"")[1]
                    print(f"Looking up Filipino word: {word}")
                    print()
                    results = get_tagalog_definition_2(word)
                    if results:
                        note = f"Top results for Tagalog word '{word}': "
                        for idx, res in enumerate(results, 1):
                            note += f"{idx}: {res['search_result']}: {res['definition']} "
                        memory.add_note(note, source="tool")
                    else:
                        memory.add_note(f"Could not find Filipino definition for '{word}'", source="tool")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path to dataset CSV")
    parser.add_argument("--out", required=True, help="Output File Name")
    args = parser.parse_args()
    main_loop()