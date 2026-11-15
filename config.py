import os
 
BASE_DIR = os.path.dirname(os.path.abspath(file))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
<<<<<<< HEAD
 
TRAIN_CSV = os.path.join(DATA_DIR, "municipal_training_set_1100.csv")
TEST_CSV = os.path.join(DATA_DIR, "Testing.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "municipal_model.pkl")
SLA_CSV = os.path.join(OUTPUT_DIR, "sla_targets.csv")
ISSUE_STATS_CSV = os.path.join(OUTPUT_DIR, "issue_stats.csv")
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "all_predictions.csv")
=======
>>>>>>> 3c30c0bbb02c4a38e6edac125f39b351488f326b

for d in (MODEL_DIR, OUTPUT_DIR):
os.makedirs(d, exist_ok=True)
