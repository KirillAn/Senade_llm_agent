

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import HuggingFacePipeline
import logging

def create_llama_pipeline(model_path: str) -> HuggingFacePipeline:
    """
    Создает и возвращает LLaMA pipeline.
    """
    logging.info("=== Создание LLaMA pipeline... ===")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    generation_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
       
    )
    llama_model = HuggingFacePipeline(pipeline=generation_pipe)

    return llama_model
