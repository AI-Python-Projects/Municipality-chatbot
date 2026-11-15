import os
 
BASE_DIR = os.path.dirname(os.path.abspath(file))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

for d in (MODEL_DIR, OUTPUT_DIR):
os.makedirs(d, exist_ok=True)
