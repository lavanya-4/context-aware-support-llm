from huggingface_hub import snapshot_download
snapshot_download("unsloth/llama-3-8b-bnb-4bit", local_dir="./llama-3-8b-bnb-4bit")