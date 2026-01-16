from google import genai
from dotenv import load_dotenv
import os
import textwrap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        use_idf=True,
        norm='l2',
        ngram_range=(1, 2),  # Use unigrams and bigrams
        sublinear_tf=True,   # Apply sublinear TF scaling
        analyzer='word'      # You could also experiment with 'char' or 'char_wb' for character-level features
    )
    tfidf = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return similarity[0][0]


def find_best_match(text_input, records):
    best_score = 0
    best_record = None
    for record in records:
        current_score = calculate_cosine_similarity(text_input, record)
        if current_score > best_score:
            best_score = current_score
            best_record = record
    return best_score, best_record


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

    temp = kwargs.get("temperature", 0.7)
    sys_inst = kwargs.get("system_instruction", "You are a helpful assistant.")

    # Перевірка: якщо іtext вже рядок, не треба робити join
    if isinstance(itext, (list, tuple)):
        text_input = "\n".join(itext)
    else:
        text_input = itext

    prompt = f"Please elaborate on the following content:\n{text_input}"
    print(prompt)

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
    "You are a specialized assistant for a textile recycling factory.",
]

# Список питань до моделі
list_query = ["What store offers rags?", "define a rag store"]
query = list_query[1]

db_records = [
    "Retrieval Augmented Generation (RAG) represents a sophisticated hybrid approach in the field of artificial intelligence, particularly within the realm of natural language processing (NLP).",
    "It innovatively combines the capabilities of neural network-based language models with retrieval systems to enhance the generation of text, making it more accurate, informative, and contextually relevant.",
    "This methodology leverages the strengths of both generative and retrieval architectures to tackle complex tasks that require not only linguistic fluency but also factual correctness and depth of knowledge.",
    "At the core of Retrieval Augmented Generation (RAG) is a generative model, typically a transformer-based neural network, similar to those used in models like GPT (Generative Pre-trained Transformer) or BERT (Bidirectional Encoder Representations from Transformers).",
    "This component is responsible for producing coherent and contextually appropriate language outputs based on a mixture of input prompts and additional information fetched by the retrieval component.",
    "Complementing the language model is the retrieval system, which is usually built on a database of documents or a corpus of texts.",
    "This system uses techniques from information retrieval to find and fetch documents that are relevant to the input query or prompt.",
    "The mechanism of relevance determination can range from simple keyword matching to more complex semantic search algorithms which interpret the meaning behind the query to find the best matches.",
    "This component merges the outputs from the language model and the retrieval system.",
    "It effectively synthesizes the raw data fetched by the retrieval system into the generative process of the language model.",
    "The integrator ensures that the information from the retrieval system is seamlessly incorporated into the final text output, enhancing the model's ability to generate responses that are not only fluent and grammatically correct but also rich in factual details and context-specific nuances.",
    "When a query or prompt is received, the system first processes it to understand the requirement or the context.",
    "Based on the processed query, the retrieval system searches through its database to find relevant documents or information snippets.",
    "This retrieval is guided by the similarity of content in the documents to the query, which can be determined through various techniques like vector embeddings or semantic similarity measures.",
    "The retrieved documents are then fed into the language model.",
    "In some implementations, this integration happens at the token level, where the model can access and incorporate specific pieces of information from the retrieved texts dynamically as it generates each part of the response.",
    "The language model, now augmented with direct access to retrieved information, generates a response.",
    "This response is not only influenced by the training of the model but also by the specific facts and details contained in the retrieved documents, making it more tailored and accurate.",
    "By directly incorporating information from external sources, Retrieval Augmented Generation (RAG) models can produce responses that are more factual and relevant to the given query.",
    "This is particularly useful in domains like medical advice, technical support, and other areas where precision and up-to-date knowledge are crucial.",
    "Retrieval Augmented Generation (RAG) systems can dynamically adapt to new information since they retrieve data in real-time from their databases.",
    "This allows them to remain current with the latest knowledge and trends without needing frequent retraining.",
    "With access to a wide range of documents, Retrieval Augmented Generation (RAG) systems can provide detailed and nuanced answers that a standalone language model might not be capable of generating based solely on its pre-trained knowledge.",
    "While Retrieval Augmented Generation (RAG) offers substantial benefits, it also comes with its challenges.",
    "These include the complexity of integrating retrieval and generation systems, the computational overhead associated with real-time data retrieval, and the need for maintaining a large, up-to-date, and high-quality database of retrievable texts.",
    "Furthermore, ensuring the relevance and accuracy of the retrieved information remains a significant challenge, as does managing the potential for introducing biases or errors from the external sources.",
    "In summary, Retrieval Augmented Generation represents a significant advancement in the field of artificial intelligence, merging the best of retrieval-based and generative technologies to create systems that not only understand and generate natural language but also deeply comprehend and utilize the vast amounts of information available in textual form.",
    "A RAG vector store is a database or dataset that contains vectorized data points."
]


best_similarity_score, best_matching_record = find_best_match(query, db_records)
print_formatted_response(best_matching_record)
print(f"Best Cosine Similarity Score: {best_similarity_score:.3f}")

# Генерація
##augmented_input = query + ": " + best_matching_record  
##llm_response = call_llm_with_full_text(
##    augmented_input , system_instruction=system_instruction[1], temperature=0.5
##)
##
##formatted_text = print_formatted_response(llm_response)
##
##with open("./test_out/keyword_matcher.txt", "w", encoding="utf-8") as f:
##    f.write(formatted_text)
