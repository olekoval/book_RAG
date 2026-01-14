from google import genai
from dotenv import load_dotenv
import os
import textwrap

from src.app.generator import RAGGenerator

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
# Генерація
client = genai.Client(api_key=api_key)
generate = RAGGenerator(client=client)

augmented_input = ("define a rag store :A RAG vector"
"store is a database or dataset that contains vectorized data points.")

prompt_used, response_text = generate.call_llm_with_full_text(augmented_input)
print(prompt_used)
print_formatted_response(response_text)
