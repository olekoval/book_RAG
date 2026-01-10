from google import genai
from dotenv import load_dotenv
import os
import textwrap

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


def call_llm_with_full_text(itext, **kwargs):
    """
    Надсилає текстовий запит до моделі Google Gemini та повертає
    згенеровану відповідь.

    Функція обробляє вхідний текст (рядок або список), формує
    промпт для розширеного
    пояснення контенту та викликає API з налаштуваннями температури
    та системних інструкцій.

    Args:
        itext (str або list): Основний текст запиту. Може бути як
        одиночним рядком, так і списком рядків (які будуть об'єднані
        через символ нового рядка).

        **kwargs: Довільні іменовані аргументи для налаштування моделі:
            - temperature (float): Рівень креативності моделі
              (за замовчуванням 0.7).
            - system_instruction (str): Системна роль або інструкція
            для моделі (за замовчуванням "You are a helpful assistant.").

    Returns:
        str: Текст відповіді від моделі або повідомлення про помилку у
             разі невдалого запиту.

    Example:
        >>> response = call_llm_with_full_text("Аналіз даних НСЗУ",
            temperature=0.5)
        >>> print(response)
    """
    # Отримуємо значення з kwargs, вказуємо 0.7 як значення за замовчуванням
    temp = kwargs.get("temperature", 0.7)
    # Отримуємо system_instruction з kwargs
    sys_inst = kwargs.get("system_instruction", "You are a helpful assistant.")

    # Перевірка: якщо іtext вже рядок, не треба робити join
    if isinstance(itext, (list, tuple)):
        text_input = "\n".join(itext)
    else:
        text_input = itext

    prompt = f"Please elaborate on the following content:\n{text_input}"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config={
                "system_instruction": sys_inst,
                "temperature": temp,
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


# Список варіантів призначення ролі моделі
system_instruction = [
    (
        "You are an expert Natural Language Processing "
        "exercise expert. You can explain the input and "
        "answer in detail."
    ),
    "You are a helpful assistant.",
    "You are a specialized " "assistant for a textile recycling factory.",
]

query = "define a rag store"

llm_response = call_llm_with_full_text(
    query #, system_instruction=system_instruction[1], temperature=0.5
)
formatted_text = print_formatted_response(llm_response)

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(formatted_text)
