from google import genai
from dotenv import load_dotenv
import os
import textwrap
from pathlib import Path

from src.app.generator import RAGGenerator
from src.app.retriever import KeywordSearch
from src.app.metrics import CosineSimilarity, EnhancedSimilarity


def print_formatted_response(response):
    # Определяем ширину абзаца для переноса текста
    wrapper = textwrap.TextWrapper(width=80)
    wrapped_text = wrapper.fill(text=response)
    print("Response:")
    print("---------------")
    print(wrapped_text)
    print("---------------\n")
    
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Считування документа контексту до списки
BASE_DIR = Path(__file__).resolve().parent # шлях до каталогу з main.py
data_path = BASE_DIR / "data" / "db_records.txt"
with open(data_path, encoding="utf-8") as f:
    db_records = [line.strip() for line in f]
    
# --- Пошук контексту до query ---
retriever = KeywordSearch(db_records)
query = "define a rag store"
score, best_matching_record = retriever.find_best_match_keyword_search(query)
augmented_input = query + ": " + best_matching_record

# --- Генерація ---
generate = RAGGenerator(client=client)
prompt_used, response_text = generate.call_llm_with_full_text(augmented_input)

# --- Метрики ---
cosine = CosineSimilarity()
# Косинусное сходство
score_cosine = cosine.calculate_cosine_similarity(query, best_matching_record)
# Розширене сходство
cosine_enh = EnhancedSimilarity()
similarity_score = cosine_enh.calculate_enhanced_similarity(
    query, best_matching_record)


# --- Друк результату ---
print(f"Best Keyword Score: {score:.3f}")
print_formatted_response(best_matching_record)
print(f"Best Cosine Similarity Score: {score_cosine:.3f}")
print(f"{query} : {best_matching_record}")
print(f"Enhanced Similarity:, {similarity_score:.3f}")
##print_formatted_response(response_text)


