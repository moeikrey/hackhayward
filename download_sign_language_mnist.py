import requests
import zipfile
import os

def download_and_extract(url, extract_to='.'):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(local_filename)
    print(f"Contents of {extract_to}:")
    for root, dirs, files in os.walk(extract_to):
        for name in files:
            print(os.path.join(root, name))

if __name__ == "__main__":
    dataset_url = "https://github.com/ardamavi/Sign-Language-Digits-Dataset/archive/refs/heads/master.zip"
    extract_path = "./sign_language_mnist"
    download_and_extract(dataset_url, extract_path)
    print(f"Dataset downloaded and extracted to {extract_path}")
