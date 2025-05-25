import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
BASE_RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')

# Upload subdirectories
INFRARED_UPLOAD_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, 'infrared')
THERMAL_UPLOAD_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, 'thermal')

# Results subdirectories
DEHAZED_RESULTS_FOLDER = os.path.join(BASE_RESULTS_FOLDER, 'dehazed')
FUSED_RESULTS_FOLDER = os.path.join(BASE_RESULTS_FOLDER, 'fused')
DETECTED_RESULTS_FOLDER = os.path.join(BASE_RESULTS_FOLDER, 'detected')

# Flask configuration
class Config:
    BASE_UPLOAD_FOLDER = BASE_UPLOAD_FOLDER
    INFRARED_UPLOAD_FOLDER = INFRARED_UPLOAD_FOLDER
    THERMAL_UPLOAD_FOLDER = THERMAL_UPLOAD_FOLDER
    DEHAZED_RESULTS_FOLDER = DEHAZED_RESULTS_FOLDER
    FUSED_RESULTS_FOLDER = FUSED_RESULTS_FOLDER
    DETECTED_RESULTS_FOLDER = DETECTED_RESULTS_FOLDER

# Create all required directories
def create_directories():
    directories = [
        BASE_UPLOAD_FOLDER,
        BASE_RESULTS_FOLDER,
        INFRARED_UPLOAD_FOLDER,
        THERMAL_UPLOAD_FOLDER,
        DEHAZED_RESULTS_FOLDER,
        FUSED_RESULTS_FOLDER,
        DETECTED_RESULTS_FOLDER
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)