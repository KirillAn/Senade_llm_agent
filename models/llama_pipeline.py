
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import HuggingFacePipeline

def create_llama_pipeline():
    """
    Создаёт и возвращает LLaMA-модель обёрнутую в HuggingFacePipeline.
    """
    model_path = ".llama/checkpoints/Llama3.2-1B-Instruct"

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
        repetition_penalty=1.15,
        device=device  
    )

    llama_model = HuggingFacePipeline(pipeline=generation_pipe)
    return llama_model
