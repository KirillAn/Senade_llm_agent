
import os
import json
import tempfile
import glob
import torch
import networkx as nx

from langchain import HuggingFacePipeline
from langchain.agents import initialize_agent, AgentType, tool
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


from models.llama_pipeline import create_llama_pipeline
from retrievers.networkx_retriever import NetworkXRetriever
from retrievers.final_answer_parser import StrictFinalAnswerParser
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def init_agent():
    """
    Инициализация модели, создание графа и настройка агента.
    Возвращает объект agent, готовый к работе.
    """
    print("=== Инициализация модели... ===")
    
    # 1) pipeline LLaMA 
    llama_model = create_llama_pipeline()

    print("\n=== Подготовка данных и графа... ===")
    folder_path = "data"  # <-- папка, где лежат *.json

    
    json_files = glob.glob(os.path.join(folder_path, "*.json"))

    all_data = []
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)

    big_text = json.dumps(all_data, ensure_ascii=False, indent=2)

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as tmp:
        tmp_file_path = tmp.name
        tmp.write(big_text)

    # Загружаем и нарезаем
    loader = TextLoader(tmp_file_path, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Чанков после нарезки: {len(docs)}")

    print("=== Создаём graph + embeddings ===")
    G = nx.Graph()
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    for i, doc in enumerate(docs):
        [chunk_emb] = embeddings_model.embed_documents([doc.page_content])
        G.add_node(i, content=doc.page_content, embedding=chunk_emb)

    print(f"Всего узлов в графе: {G.number_of_nodes()}")

    print("=== Настройка агента... ===")
    graph_retriever = NetworkXRetriever(graph=G, embeddings=embeddings_model, k=3)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llama_model,
        chain_type="stuff",
        retriever=graph_retriever,
        return_source_documents=False
    )

    @tool
    def ask_ctf_knowledge(query: str) -> str:
        """Отвечает на вопросы по теме CTF, используя RetrievalQA + наш граф."""
        return rag_chain.run(query)

    tools = [ask_ctf_knowledge]

    custom_prompt = PromptTemplate.from_template("""
Question: {input}
Final Answer: {output}
""")

    agent = initialize_agent(
        tools=tools,
        llm=llama_model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        agent_kwargs={"prompt": custom_prompt},
        output_parser=StrictFinalAnswerParser()
    )

    print("=== АГЕНТ ГОТОВ ===\n")
    return agent
