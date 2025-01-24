

import os
import json
import tempfile
import glob
import logging

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from models.llama_pipeline import create_llama_pipeline
from retrievers.networkx_retriever import NetworkXRetriever

import networkx as nx

def initialize_agent() -> RetrievalQA:
    """
    Инициализирует модель, загрузчик данных, граф, эмбеддинги и цепочку RetrievalQA.
    Возвращает объект RetrievalQA.
    """
    logging.info("=== Инициализация агента... ===")
    model_path = "/Users/kirillanosov/.llama/checkpoints/Llama3.2-1B-Instruct"

    # Создаем LLaMA pipeline
    llama_model = create_llama_pipeline(model_path)

    logging.info("\n=== Подготовка данных и графа... ===")
    folder_path = "/Users/kirillanosov/Desktop/Управление проектами/json_obj"

    # Ищем все файлы *.json в указанной папке
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

    loader = TextLoader(tmp_file_path, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    logging.info(f"Чанков после нарезки: {len(docs)}")

    logging.info("=== Создаём граф и эмбеддинги... ===")
    G = nx.Graph()

    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    for i, doc in enumerate(docs):
        [chunk_emb] = embeddings_model.embed_documents([doc.page_content])
        G.add_node(i, content=doc.page_content, embedding=chunk_emb)

    logging.info(f"Всего узлов в графе: {G.number_of_nodes()}")

    logging.info("=== Настройка RetrievalQA... ===")
    graph_retriever = NetworkXRetriever(graph=G, embeddings=embeddings_model, k=3)

   
    custom_prompt = PromptTemplate.from_template("""
You are an assistant that provides concise and direct answers to questions without any additional explanations.

Context: {context}

Question: {question}
Final Answer:
""")

    rag_chain = RetrievalQA.from_chain_type(
        llm=llama_model,
        chain_type="stuff",  
        chain_type_kwargs={"prompt": custom_prompt},
        retriever=graph_retriever,
        return_source_documents=False
       
    )

    return rag_chain
