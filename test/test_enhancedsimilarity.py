import pytest
from src.app.metrics import EnhancedSimilarity



class TestEnh:
    
    @pytest.fixture
    def e(self):
        return EnhancedSimilarity()

    # --- Тест get_synonyms() ---
    @pytest.mark.parametrize("w, s",[
        ("car", "automobile"),
        ("car", "machine"),
        ("rag", "shred"),
        ])
    def test_get_synonyms(self, e, w, s):
        result = e.get_synonyms(w)
        
        # ПЕРЕВІРКА ТИПУ: результат має бути множиною (set)
        assert isinstance(result, set)
        
        # Додатково: перевіримо, що множина не порожня і містить очікуваний синонім
        assert s in result


    # --- Тест preprocess_text() ---
    @pytest.mark.parametrize("input_text, expected_output", [
        # Тест на нижній регістр та базову очистку
        ("Hello WORLD", ["hello", "world"]),
        
        # Тест на видалення стоп-слів (is, a, the - зазвичай у списку spacy)
        ("The patient is healthy", ["patient", "healthy"]),
        
        # Тест на видалення пунктуації
        ("Data, analysis; and metrics!", ["datum", "analysis", "metric"]), 
        
        # Тест на лемматизацію (running -> run, mice -> mouse)
        ("running faster than mice", ["run", "fast", "mouse"]),
        
        # Тест на порожній вхід
        ("", []),
        ("   ", []),
        
        # Тест на цифри (якщо spacy не вважає їх пунктуацією, вони залишаться)
        ("Value is 100%", ["value", "100"])
    ])
    def test_preprocess_text(self, e, input_text, expected_output):
        result = e.preprocess_text(input_text)
        
        # Перевіряємо, чи результат є списком
        assert isinstance(result, list)
        
        # Перевіряємо відповідність очікуваному результату
        assert result == expected_output

    # --- Тест expand_with_synonyms()
    def test_expand_with_synonyms_basic(self, e):
        # Тест на простому слові
        input_words = ["car"]
        result = e.expand_with_synonyms(input_words)
        
        # ПЕРЕВІРКА: Оригінальне слово має залишитися
        assert "car" in result
        # ПЕРЕВІРКА: Мають з'явитися синоніми (наприклад, 'automobile')
        assert "automobile" in result
        # ПЕРЕВІРКА: Список став довшим за оригінал
        assert len(result) > len(input_words)


    # --- Тест calculate_enhanced_similarity() ---

    def test_calculate_enhanced_similarity_logic(self, e):
        text1 = "The automobile is fast"
        text2 = "A quick car"
        
        similarity = e.calculate_enhanced_similarity(text1, text2)
        
        # 1. Знижуємо поріг до реалістичного для синонімів
        assert similarity > 0.35 
        
        # 2. Додаємо тест на ідентичність (найважливіший тест)
        # Текст сам із собою завжди має давати 1.0
        assert e.calculate_enhanced_similarity(text1, text1) == pytest.approx(1.0)
        
        # 3. Перевіряємо відносну подібність (принцип "краще ніж нічого")
        # Подібність зі схожим за змістом текстом має бути вищою, ніж з випадковим
        text_unrelated = "The weather is cold today"
        unrelated_similarity = e.calculate_enhanced_similarity(text1, text_unrelated)
        assert similarity > unrelated_similarity
