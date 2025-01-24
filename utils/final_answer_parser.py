

from langchain.schema import BaseOutputParser

class FinalAnswerParser(BaseOutputParser):
    """
    Парсер для извлечения только финального ответа.
    Поддерживает различные варианты маркеров.
    """
    def parse(self, text: str) -> str:
        final_markers = ["Final Answer:", "Answer:", "Ответ:"]
        for marker in final_markers:
            idx = text.find(marker)
            if idx != -1:
                return text[idx + len(marker):].strip()
        # Если маркер не найден, вернём весь текст
        return text.strip()
