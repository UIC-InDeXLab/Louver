"""Run on login node to pre-download all models before compute job."""
from transformers import AutoModelForCausalLM, AutoTokenizer

models = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
]

for m in models:
    print(f"Downloading {m} ...")
    AutoTokenizer.from_pretrained(m)
    AutoModelForCausalLM.from_pretrained(m)
    print(f"  done.")

print("All models cached.")
