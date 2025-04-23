import os
import requests
import hashlib
from tqdm import tqdm

MODEL_FILES = {
    'config.json': 'https://huggingface.co/hfl/chinese-roberta-wwm-ext/resolve/main/config.json',
    'pytorch_model.bin': 'https://huggingface.co/hfl/chinese-roberta-wwm-ext/resolve/main/pytorch_model.bin',
    'tokenizer_config.json': 'https://huggingface.co/hfl/chinese-roberta-wwm-ext/resolve/main/tokenizer_config.json',
    'vocab.txt': 'https://huggingface.co/hfl/chinese-roberta-wwm-ext/resolve/main/vocab.txt'
}

def download_file(url, output_path, session):
    """Download a file with progress bar."""
    response = session.get(url, stream=True, verify=False)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def main():
    # Create model directory
    model_dir = "model_cache/chinese-roberta-wwm-ext"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create a session with disabled SSL verification
    session = requests.Session()
    session.verify = False
    
    # Download each model file
    for filename, url in MODEL_FILES.items():
        output_path = os.path.join(model_dir, filename)
        if not os.path.exists(output_path):
            print(f"Downloading {filename}...")
            download_file(url, output_path, session)
        else:
            print(f"File {filename} already exists, skipping...")
    
    print("\nModel files downloaded successfully!")
    print(f"Model files are located in: {os.path.abspath(model_dir)}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
    main()