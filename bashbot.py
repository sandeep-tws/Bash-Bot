#!/usr/bin/env python3
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import warnings
warnings.filterwarnings("ignore")

# Also silence transformers / accelerate / bitsandbytes logs
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =========================================================
# CONFIGURATION
# =========================================================
MERGED_MODEL = "/home/sandeep_naruka/bashbot/Models/bashbot_merged"  # path to merged model
OFFLOAD_DIR = "/home/sandeep_naruka/bashbot/offload"

os.makedirs(OFFLOAD_DIR, exist_ok=True)

# =========================================================
# LOAD TOKENIZER
# =========================================================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# =========================================================
# LOAD MERGED MODEL
# =========================================================
print("Loading merged model...")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=False,
    llm_int8_enable_fp32_cpu_offload=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL,
    quantization_config=bnb_config,
    dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)
model.eval()

# =========================================================
# GENERATION FUNCTION
# =========================================================
def generate_command(prompt: str, max_tokens: int = 25) -> str:
    """
    Generate a bash command from the prompt using the loaded model.
    Optimized for speed and clean output.
    """
    if not prompt.strip():
        return "mpty prompt provided. Please enter a valid command request."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    with torch.inference_mode():  # faster than torch.no_grad()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,          # greedy decoding is faster
            repetition_penalty=1.2,   # reduce repeated words
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode and clean up the output
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.strip()

# =========================================================
# MAIN CLI ENTRY
# =========================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: bashbot <your request>")
        sys.exit(1)

    user_prompt = " ".join(sys.argv[1:])
    print(f"Prompt: {user_prompt}\n")
    result = generate_command(user_prompt)
    print("Bashbot Output:\n")
    print(result)
