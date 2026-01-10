from google import genai
from dotenv import load_dotenv
import os
import textwrap

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


def call_llm_with_full_text(itext):
    # Перевірка: якщо іtext вже рядок, не треба робити join
    if isinstance(itext, (list, tuple)):
        text_input = "\n".join(itext)
    else:
        text_input = itext

    prompt = f"Please elaborate on the following content:\n{text_input}"

# Варіанти призначення ролі для LLM
# Варіант 1:
##    system_instruction = (
##        "You are an expert Natural Language Processing "
##        "exercise expert. You can explain the input and "
##        "answer in detail."
##    )

# Варіант 2:
    system_instruction = "You are a helpful assistant."

# Варіант 3:
##    system_instruction = (
##        "You are a specialized assistant for a textile recycling factory."
##     )
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config={
                "system_instruction": system_instruction,
                "temperature": 0.99,
            },
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Помилка при виклику API: {e}"


def print_formatted_response(response):
    wrapper = textwrap.TextWrapper(width=80)
    wrapped_text = wrapper.fill(text=response)

    print("Response:")
    print("---------------")
    print(wrapped_text)
    print("---------------\n")
    
    return wrapped_text


query = "define a rag store"

llm_response = call_llm_with_full_text(query)
formatted_text = print_formatted_response(llm_response)

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(formatted_text)
