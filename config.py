import os
 
BASE_DIR = os.path.dirname(os.path.abspath(file))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
 
TRAIN_CSV = os.path.join(DATA_DIR, "municipal_training_set_1100.csv")
TEST_CSV = os.path.join(DATA_DIR, "Testing.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "municipal_model.pkl")
SLA_CSV = os.path.join(OUTPUT_DIR, "sla_targets.csv")
ISSUE_STATS_CSV = os.path.join(OUTPUT_DIR, "issue_stats.csv")
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "all_predictions.csv")

for d in (MODEL_DIR, OUTPUT_DIR):
os.makedirs(d, exist_ok=True)

COL_DESC = "issue_description"
COL_TYPE = "issue_type"
COL_DEPT = "municipal_department"
COL_PRIORITY = "priority_level"
COL_SEVERITY = "severity_score"
COL_AREA = "area_importance"
COL_REPORTS = "citizen_reports_count"
COL_ANON = "is_anonymous"
COL_REPORT_DT = "report_datetime"
COL_RESOLVE_DT = "resolution_datetime"
COL_STATUS = "issue_status"
COL_CITY = "location_city"
COL_STATE = "location_state"

W_SEVERITY = 0.40
W_AREA = 0.30
W_REPORTS = 0.60

PRIORITY_BANDS = {
"critical": 8.5,
"high": 7.0,
"medium": 4.0,

}
SLA_BUFFER        = 1.15
DEFAULT_SLA_HOURS = {
"critical": 8,
"high": 24,
"medium": 72,
"low": 168
}
MIN_SAMPLES_PER_CLASS = 5

