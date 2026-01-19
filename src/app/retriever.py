from src.app.metrics import CosineSimilarity

class KeywordSearch:
    """Пошук контексту до заптання по ключовим словам"""

    def __init__(self, db_records):
        self.db_records = db_records

    def find_best_match_keyword_search(self, query):
        best_score = 0
        best_record = None

        # Split the query into individual keywords
        query_keywords = set(query.lower().split())

        # Iterate through each record in db_records
        for record in self.db_records:
            # Split the record into keywords
            record_keywords = set(record.lower().split())

            # Calculate the number of common keywords
            common_keywords = query_keywords.intersection(record_keywords)
            current_score = len(common_keywords)

            # Update the best score and record if the current score is higher
            if current_score > best_score:
                best_score = current_score
                best_record = record

        return best_score, best_record

class VectorSearch:
    """Векторний пошук"""
    def __init__(self, db_records):
        self.db_records = db_records
        self.cos = CosineSimilarity()

    def find_best_match(self, text_input, records):
        best_score = 0
        best_record = None
        for record in records:
            current_score = self.cos.calculate_cosine_similarity(text_input,
                                                        record)
            if current_score > best_score:
                best_score = current_score
                best_record = record
        return best_score, best_record
    


if __name__ == "__main__":

    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
    data_path = BASE_DIR / "data" / "db_records.txt"

    p = Path(__file__).resolve()

    print(p)  # retriever.py
    print(p.parent)  # app
    print(p.parents[1])  # src
    print(p.parents[2])  # book_RAG

    with open(data_path, encoding="utf-8") as f:
        db_records = [line.strip() for line in f]

    retriever = KeywordSearch(db_records)
    query = "define a rag store"

    score, top_record = retriever.find_best_match_keyword_search(query)
    print(score, top_record)
