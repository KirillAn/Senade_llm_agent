# Senade_llm_agent

# Проект: Атакующий LLM-агент для Capture The Flag (CTF) Senade
```
senade_llm_agent/
├─ app.py               # Входная точка (main, Gradio, запуск агента)
├─ agents/
│   └─ ctf_agent.py     # init_agent, настройка ZeroShotReact, tools
├─ retrievers/
│   └─ networkx_retriever.py  # класс NetworkXRetriever
├─ models/
│   └─ llama_pipeline.py # создание LLaMA pipeline
├─ data/
│   └─ (json файлы, txt файлы, etc.)
├─ utils/
│   └─ final_answer_parser.py # StrictFinalAnswerParser
└─ requirements.txt
```

#### Логика работы
```mermaid
flowchart TD
    %% Схема сверху вниз (TD)

    %% Основные блоки
    documents[Source Documents JSON]:::yellowBox
    createEmbeddings[Create Embeddings]:::blueBox
    insertDB[Insert into Graph DB ]:::orangeBox
    question[Ask Question]:::greenBox
    LLM[LLM ]:::purpleBox
    answer[Generate Answer]:::purpleBox

    %% Блоки для поиска
    search[Search Relevant Documents]:::grayBox
    retriever[Retriever Graph DB]:::grayBox

    %% Связи
    documents --> createEmbeddings
    createEmbeddings --> insertDB
    insertDB --> retriever
    question --> LLM
    LLM --> answer
    LLM --> search
    search --> retriever

    %% Стили блоков
    classDef yellowBox fill:#fff3b3,color:#333,stroke:#ffd966,stroke-width:2px
    classDef greenBox fill:#dafbe1,color:#333,stroke:#8dde98,stroke-width:2px
    classDef blueBox fill:#d4efff,color:#333,stroke:#5dc8f4,stroke-width:2px
    classDef purpleBox fill:#fce4ff,color:#333,stroke:#fcb0ff,stroke-width:2px
    classDef orangeBox fill:#ffe5d1,color:#333,stroke:#ffa45b,stroke-width:2px
    classDef grayBox fill:#f4f4f4,color:#333,stroke:#ccc,stroke-width:2px
```

#### Пример

<pre>
  <div id="header" align="center"> <img src=data/ezgif.com-video-to-gif-converter.gif width="1000"/>
  </div>
</pre>

## 🧠 Атакующий LLM-агент

Проект представляет собой разработку атакующего агента на базе языковой модели (LLM), специально предназначенного для решения задач в формате Capture The Flag (CTF). Атакующий агент будет анализировать систему, выявлять уязвимости и генерировать команды для эксплуатации этих уязвимостей, что позволит моделировать реальные сценарии пентестинга и улучшать защитные стратегии.

### 🎯 Цели проекта

- Создать агент на основе LLM, который будет способен анализировать текстовые данные и логи систем для поиска уязвимостей.
- Автоматизировать процесс генерации атакующих команд, эксплойтов и действий для выполнения CTF-задач.
- Моделировать автономные атаки в симулированной инфраструктуре с использованием наборов данных CTF.
- Оптимизировать тренировки пентестеров и тестирование систем безопасности.

### 🔑 Ключевые функции

1. **Анализ уязвимостей:**
   - Обучение модели для выявления паттернов уязвимостей в текстовых логах, системных сообщениях и сетевых данных.
   - Определение точек повышения привилегий и системных ошибок.

2. **Генерация атакующих команд:**
   - Создание атакующих скриптов и команд на основе анализа входных данных.
   - Автоматическая генерация эксплойтов для выполнения атак CTF, включая SQL-инъекции, обход аутентификации и повышение привилегий.

3. **Автономное выполнение атак:**
   - Автоматизация процесса выполнения атак в симулированной ИТ-среде (виртуальные машины, контейнеры).
   - Оценка успешности атак с возможностью обратной связи и обучения модели.

4. **Адаптация к новым сценариям:**
   - Адаптивное обучение на основе результатов атак, корректировка стратегии в зависимости от поведения защищаемой системы.
   - Возможность добавления новых типов атак и уязвимостей через обучение модели на дополнительных данных.

### 🛠 Технологии

- LLM: llama3.1-8b-instruct +  llama3.2-1b-instruct
- Фреймворк: Gradio
- База данных: граф NetworkX


  #### Архитектурная цепочка
```mermaid
flowchart LR
    %% Говорим, что схема идёт слева направо (LR)
    
    subgraph GRADIO
    UI[(Gradio Interface)]:::greenBox
    askAgent((ask_agent)):::method
    UI --> askAgent
    end

    subgraph AGENT
    AG[(ZeroShotReAct Agent)]:::purpleBox
    agentInit((initialize_agent)):::method
    AG --> agentInit
    end

    subgraph RAG
    RAGChain[(RetrievalQA\nRAG Chain)]:::blueBox
    ragRun((run)):::method
    RAGChain --> ragRun
    end

    subgraph RETRIEVER
    NXRet[(NetworkXRetriever)]:::grayBox
    nxDocs((get_relevant_documents)):::method
    NXRet --> nxDocs
    end

    subgraph GRAPH
    NxG[(networkx.Graph)]:::grayBox
    storeEmb((store chunks + embeddings)):::method
    NxG --> storeEmb
    end

    subgraph LLM
    HF[(HuggingFacePipeline)]:::blueBox
    hfGen((generate)):::method
    HF --> hfGen
    end

    %% Допустим, есть Tool (ask_ctf_knowledge)
    Tool[(Tool:\nask_ctf_knowledge)]:::grayBox

    %% Связи между блоками
    UI --> AG
    AG --> RAGChain
    RAGChain --> NXRet
    NXRet --> NxG
    RAGChain --> HF
    AG --> Tool

    %% Опционально оформим стили
    classDef greenBox fill:#dafbe1,color:#333,stroke:#8dde98,stroke-width:2px
    classDef purpleBox fill:#fce4ff,color:#333,stroke:#fcb0ff,stroke-width:2px
    classDef blueBox fill:#d4efff,color:#333,stroke:#5dc8f4,stroke-width:2px
    classDef grayBox fill:#f4f4f4,color:#333,stroke:#ccc,stroke-width:2px
    classDef method fill:#fff,color:#333,stroke:#999,stroke-width:1px,stroke-dasharray:3 2

```

### 🚀 TODO

- Интеграция с внешними инструментами: анализатор кода, портов и тд
- Подключение более сильных моделей - GPT, Claude и тд.
- Обогащение датасета




---

Создано для того, чтобы помочь пентестерам и специалистам по безопасности моделировать и изучать реальные сценарии атак.
