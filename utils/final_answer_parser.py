
from langchain.schema import BaseOutputParser

class StrictFinalAnswerParser(BaseOutputParser):
    """
    Парсер для извлечения только финального ответа.
    """
    def parse(self, text: str) -> str:
        final_marker = "Final Answer:"
        idx = text.find(final_marker)
        if idx == -1:
            return text.strip()
        return text[idx + len(final_marker):].strip()
