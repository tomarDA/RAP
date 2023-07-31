import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

model_name = "decapoda-research/llama-7b-hf"

model = LlamaForCausalLM.from_pretrained(model_name, load_in_8bit = True, torch_dtype = torch.float16,
                                         device_map = "auto")

tokenizer = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token