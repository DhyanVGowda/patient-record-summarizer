import os
import subprocess

MODEL_ID = "1hXs0uG8b6mvA9BcDm5zyW2EfvnNdduKN"
MODEL_FILENAME = "en_ner_bc5cdr_md-0.5.3.tar.gz"

def download_and_install_model():
    if not os.path.exists(MODEL_FILENAME):
        print("Downloading the SciSpacy model...")
        subprocess.run(["gdown", f"https://drive.google.com/uc?id={MODEL_ID}"], check=True)
    else:
        print("Model archive already exists. Skipping download.")

    print("Installing the SciSpacy model...")
    subprocess.run(["pip", "install", MODEL_FILENAME], check=True)

if __name__ == "__main__":
    download_and_install_model()