import os
import torch
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load environment variables from .env file
load_dotenv(override=True)

# Get the token and check if it exists
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables or .env file!")

print(f"Token loaded: {'✓' if hf_token else '✗'}")

# Login to Hugging Face
login(token=hf_token)

# First check what directories are available for storage
os.system("df -h")

# Create a directory in your home folder instead
model_path = Path.home() / "mistral_models" / "7B-v0.3"
model_path.mkdir(parents=True, exist_ok=True)
print(f"Using model directory: {model_path}")

# Configure quantization
bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.float32
)

# Download model files
print("Downloading model files...")
snapshot_download(
    repo_id="mistralai/Mistral-7B-v0.3", 
    local_dir=model_path,
    token=hf_token
)

# Load the model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda:0", 
    quantization_config=bnb_config
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Model loaded successfully!")
