import os
import glob 
import torch
from ultralytics import YOLO
from src.object_detector import make_coercion_check
from src.results_handler import ResultsHandler
import yaml
import argparse
import logging
import sys


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_logging(output_path=None, log_level="INFO"):
    """
    Set up logging to both console and file if output_path is provided.
    
    Args:
        output_path (str): Directory where the log file will be saved
        log_level (str): Logging level to set for the logger
    """
    # Configure the root logger
    log_level_obj = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(log_level_obj)
    root_logger.addHandler(console_handler)
    
    # Add file handler if output path is provided
    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        log_file = os.path.join(output_path, "coercion_check_log.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level_obj)
        root_logger.addHandler(file_handler)
        
        logging.info(f"Logging to file: {log_file}")
    
    return root_logger

# Parse arguments
parser = argparse.ArgumentParser(description="Run coercion detection on video files.")
parser.add_argument(
    "--config", 
    type=str, 
    default="configs/default_config.yaml", 
    help="Path to the configuration YAML file"
)

args = parser.parse_args()
config_path = args.config

# Load configuration from YAML file
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

YOLO_MODEL_PATH = config.get("YOLO_MODEL_PATH", os.path.join("src", "models", "yolo_model8m.pt"))
CONFIDENCE_THRESHOLD = config.get("CONFIDENCE_THRESHOLD", 0.6)
VIDEO_PATHS = config.get("VIDEO_PATHS", os.path.join("videos", "*.mp4"))
OUTPUT_DIR = config.get("OUTPUT_DIR", "output")
LOG_LEVEL = config.get("LOG_LEVEL", "INFO")
VISUALIZATION = config.get("VISUALIZATION", True)

if isinstance(VIDEO_PATHS, str):
    video_paths = glob.glob(VIDEO_PATHS)
else:
    video_paths = VIDEO_PATHS

if __name__ == "__main__":
    
    # Initialize results handler
    results_handler = ResultsHandler(OUTPUT_DIR)
    output_path = results_handler.output_dir

    # Set up logging to both console and file
    setup_logging(output_path, LOG_LEVEL)
    logging.info(f"Starting coercion detection on {len(video_paths)} videos")
    logging.info(f"Output directory: {output_path}")

    # Load model
    logging.info(f"Loading YOLO model from {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)

    # Check if CUDA is available and use GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    model = model.to(device)

    
    # Run coercion detection
    logging.info(f"Running coercion detection with confidence threshold: {CONFIDENCE_THRESHOLD}")
    coercion_results = make_coercion_check(
        video_paths, 
        model,
        conf_threshold=CONFIDENCE_THRESHOLD,
        save_visualizations=VISUALIZATION,
        output_path=output_path
    )

    # Output results as log and print to console
    logging.info("Saving detection results...")
    results_handler.save_all_results(coercion_results) 
    logging.info("Coercion detection completed successfully.")