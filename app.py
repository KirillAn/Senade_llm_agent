
import gradio as gr
from agents.ctf_agent import init_agent

def main():
    # Шаг 1: Инициализируем агента
    agent = init_agent()

    # Шаг 2: Определяем функцию для Gradio
    def ask_agent(user_input):
        try:
            result = agent.run(user_input)
            return result
        except Exception as e:
            return f"Error: {e}"

    # Шаг 3: Gradio-интерфейс
    demo = gr.Interface(
        fn=ask_agent,
        inputs="text",
        outputs="text",
        title="CTF Graph Assistant",
        description="Задайте вопрос по теме CTF"
    )

    # Запуск 
    demo.launch(server_name="0.0.0.0", server_port=7861, open_browser=True)

if __name__ == "__main__":
    main()
