"""
Клас для взаємодії з Google Gemini API в межах RAG-системи.
Забезпечує формування промптів та генерацію відповідей на основі
наданого контексту.

Attributes:
    model (str): Назва моделі Gemini, яка використовується для генерації.
"""
from typing import Union, List, Tuple  # Додайте цей імпорт
    
class RAGGenerator:

    def __init__(self, client, model: str = "gemini-2.5-flash"):
        self.client = client
        self.model = model
        """
        Ініціалізує екземпляр RAGGenerator.

        Args:
            model (str): Ідентифікатор моделі (напр., "gemini-2.5-flash"). 
        """
    def call_llm_with_full_text(
        self,
        itext: Union[str, List[str], Tuple[str, ...]],
        system_instruction: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        beginning_prompt: str = "Please elaborate on the following content:",
    )-> Tuple[str, str]:
        """
        Відправляє запит до LLM, об'єднуючи контекст та інструкції.
        Цей метод готує фінальний промпт, викликає API моделі через глобальний 
        об'єкт `client` та обробляє можливі виключення.

        Args:
            itext (Union[str, List[str], Tuple[str, ...]]): Основний текст або 
                список фрагментів тексту для аналізу.
            system_instruction (str): Системна установка, що визначає роль моделі.
            temperature (float): Параметр креативності моделі (від 0.0 до 2.0).
            beginning_prompt (str): Вступна фраза перед основним текстом.

        Returns:
            Tuple[str, str]: Кортеж, що містить:
                - Сформований повний промпт (для логування/аналітики).
                - Текст відповіді від моделі або повідомлення про помилку.

        Raises:
            Exception: Будь-які помилки API перехоплюються і повертаються як рядок.
        """
        # Перевірка: якщо іtext вже рядок, не треба робити join
        if isinstance(itext, (list, tuple)):
            text_input = "\n".join(itext)
        else:
            text_input = itext

        prompt = (f"{beginning_prompt}\n"
                  f"--- START OF CONTENT ---\n"
                  f"{text_input}\n"
                  f"--- END OF CONTENT ---"
                  )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                config={
                    "system_instruction": system_instruction,
                    "temperature": temperature,
                },
                contents=prompt,
            )
            return prompt, response.text
        except Exception as e:
            return prompt, f"Помилка при виклику API: {e}"
