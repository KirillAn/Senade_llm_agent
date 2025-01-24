# app.py

import gradio as gr
import logging

from agents.ctf_agent import initialize_agent
from utils.final_answer_parser import FinalAnswerParser

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Инициализируем цепочку RetrievalQA один раз при запуске
rag_chain = initialize_agent()

def ask_agent(question: str) -> str:
    """
    Функция для обработки запроса пользователя и получения ответа от цепочки RetrievalQA.
    """
    try:
        logging.info(f"Получен вопрос: {question}")
        raw_answer = rag_chain.run(question)
        logging.info(f"Сгенерированный ответ (raw): {raw_answer}")

        # Применяем парсер для извлечения финального ответа
        parser = FinalAnswerParser()
        final_answer = parser.parse(raw_answer)
        logging.info(f"Сгенерированный ответ (parsed): {final_answer}")

        return final_answer
    except Exception as e:
        logging.error(f"Ошибка в ask_agent: {e}")
        return f"Ошибка: {e}"

def respond(message, history):
    """
    Обрабатывает сообщение пользователя, генерирует ответ и обновляет историю чата.
    """
    history = history or []
    history.append((message, "Генерация ответа..."))
    answer = ask_agent(message)
    history[-1] = (message, answer)
    return history, history

# Настройка интерфейса Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Chat with a CTF Assistant 🦜⛓️")

    chatbot = gr.Chatbot(label="CTF Assistant")

    with gr.Row():
        input_box = gr.Textbox(
            lines=1,
            label="Chat Message",
            placeholder="Type your message here and press Enter..."
        )

    state = gr.State([])  # Инициализация состояния истории чата

    input_box.submit(respond, [input_box, state], [chatbot, state])
    input_box.submit(lambda: "", None, input_box)  # Очистка текстового поля после отправки

# Запуск интерфейса с использованием системы очередей Gradio
if __name__ == "__main__":
    demo.queue()  # Инициализация очереди
    demo.launch(share=True)
