# Fine-Tuning Llama 2 7B Chat with QLoRA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1poa5lHeaSSdMy-jJKXZp396y64wNdZ0B?usp=sharing)
[![Model on HuggingFace](https://img.shields.io/badge/Model-pkumar02%2FLlama--2--7b--chat--finetune-blue)](https://huggingface.co/pkumar02/Llama-2-7b-chat-finetune)
[![Dataset](https://img.shields.io/badge/Dataset-guanaco--llama2--1k-green)](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k)

Fine-tuning Meta's Llama 2 7B Chat with QLoRA (4-bit + LoRA) on a single GPU via Colab.

## Quick Links

- Model: https://huggingface.co/pkumar02/Llama-2-7b-chat-finetune
- Colab: https://colab.research.google.com/drive/1poa5lHeaSSdMy-jJKXZp396y64wNdZ0B?usp=sharing

## Repo Structure

```
Fine_tune_Llama_2_7b_chat.ipynb
Guanaco_Llama_Dataset_fixed.ipynb
README.md
```

## What Is Inside

- `Guanaco_Llama_Dataset_fixed.ipynb`: Samples 1,000 rows from `timdettmers/openassistant-guanaco`, reformats them into the Llama 2 chat template, and pushes to the Hub.
- `Fine_tune_Llama_2_7b_chat.ipynb`: QLoRA fine-tuning on `mlabonne/guanaco-llama2-1k`, merges LoRA weights, and pushes the final model.

Template used for the dataset:

```
<s>[INST] {user_instruction} [/INST] {model_response} </s>
```

## Inference (Example)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_name = "pkumar02/Llama-2-7b-chat-finetune"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=200)
prompt = "<s>[INST] What is a large language model? [/INST]"
print(pipe(prompt)[0]["generated_text"])
```

## Notes

- QLoRA enables 7B fine-tuning on free-tier GPUs by loading the base model in 4-bit (NF4) and training LoRA adapters.
- The notebook merges LoRA weights into the base model before pushing to the Hub for easy inference.
