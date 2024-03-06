import requests, gzip, shutil
from pathlib import Path

def download_and_extract(url, dest_path):
    """Download and extract gz files."""
    filename = url.split('/')[-1]
    filepath = dest_path / filename
    
    # Download the file
    print(f"Downloading {filename}...")
    response = requests.get(url)
    filepath.write_bytes(response.content)
    
    # Extract the file
    print(f"Extracting {filename}...")
    with gzip.open(filepath, 'rb') as f_in:
        with open(filepath.with_suffix(''), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Remove the .gz file
    filepath.unlink()

def main():
    DATA_DIR = Path(__file__).resolve().parent.parent.joinpath("examples").joinpath("data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    urls = [
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    ]

    for url in urls:
        download_and_extract(url, DATA_DIR)

if __name__ == "__main__":
    main()
