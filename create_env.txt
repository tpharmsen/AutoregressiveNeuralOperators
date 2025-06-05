conda create -n grad311 python=3.11 -y
conda activate grad311
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt