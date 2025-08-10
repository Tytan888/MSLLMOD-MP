import os
import json
import time
import requests

from transformers import AutoTokenizer, AutoModel
import torch

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup

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

        time.sleep(3)  # wait for JS & Cloudflare

        soup = BeautifulSoup(driver.page_source, "html.parser")

        results = []

        for i in range(1, 3):
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
            "accuracy_weight": "High",
            "fluency_weight": "High",
            "completeness_weight": "Low",
            "cultural_appropriateness_weight": "Low",
            "user_notes": [],
            "tool_notes": [],
            "past_questions": [],
            "has_asked_for_additional_instructions": False,
            "has_asked_if_weights_are_acceptable": False,
            "english_words_queried": [],
            "filipino_words_queried": [],
            "similarities_queried": []
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

    def add_english_word(self, word):
        """Add an English word to the memory."""
        self.memory["english_words_queried"].append(word)

    def add_filipino_word(self, word):
        """Add a Filipino word to the memory."""
        self.memory["filipino_words_queried"].append(word)

    def add_similarity(self, similarity):
        """Add a similarity score to the memory."""
        self.memory["similarities_queried"].append(similarity)

    def __str__(self):
        return json.dumps(self.memory, ensure_ascii=False, indent=2)
