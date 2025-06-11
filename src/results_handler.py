import os
import datetime
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ResultsHandler:
    def __init__(self, output_parent_dir="output"):
        self.set_output_directory(output_parent_dir)

    def set_output_directory(self, OUTPUT_DIR):
        """Get the output directory for results."""
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_dir = os.path.join(OUTPUT_DIR, current_datetime)
        self.output_dir = output_dir

    def create_output_directory(self):
        """Create a directory for results using the current date."""
        # Create directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logging.info(f"Created output directory: {self.output_dir}")

    def save_all_results(self, coercion_results):
        self.save_coercion_results(coercion_results)
        self.save_detailed_results(coercion_results)
        self.save_summary(coercion_results)

    def save_summary(self, coercion_results):
        """Create and save a summary report."""
        summary_path = os.path.join(self.output_dir, "summary.txt")

        with open(summary_path, "w") as f:
            f.write("Coercion Detection Results Summary\n")
            f.write("=================================\n\n")
            
            # Basic statistics
            total_videos = coercion_results['video'].nunique()
            total_persons = coercion_results['person_id'].nunique()
            f.write(f"Total videos processed: {total_videos}\n")
            f.write(f"Total persons detected: {total_persons}\n\n")

            f.write("=================================\n\n")

            # Coercion detection statistics
            coercion_videos = coercion_results[coercion_results['coercion_detected'] == 1]['video'].unique()

            f.write(f"Total coercion detected: {len(coercion_videos)} out of {total_videos} ({(len(coercion_videos) / total_videos) * 100:.2f}%)\n\n")
            # Videos with coercion detected
            f.write(f"Videos with coercion detected ({len(coercion_videos)}):\n")
            for video in coercion_videos:
                persons_detected = coercion_results[coercion_results['video'] == video]['person_id'].nunique()
                f.write(f"- {video}: {persons_detected} persons detected\n")
            f.write("\n")

        logging.info(f"Saved summary report to {summary_path}")

    def save_coercion_results(self, coercion_results):
        """
        Save coercion detection results to files in a dated output directory.
        
        Args:
            coercion_results (pd.DataFrame): DataFrame containing detection results
            confidence_threshold (float): The confidence threshold used for detection
            
        Returns:
            pd.DataFrame: The processed results DataFrame
        """
        
        # Generate aggregated results by video
        if "video" in coercion_results.columns:
            final_results = pd.DataFrame(coercion_results.groupby("video")['coercion_detected'].mean())
            final_results = final_results.reset_index()
            final_results.columns = ['video', 'coercion_detected']
            final_results['coercion_detected'] = (final_results['coercion_detected'] > 0).astype(int)
        else:
            final_results = coercion_results
        
        # Log final results
        logging.info(f"Coercion detection results:\n{final_results}")

        # Save aggregated results
        results_path = os.path.join(self.output_dir, "results.csv")
        final_results.to_csv(results_path, sep = ";", index=False)
        logging.info(f"Saved aggregated results to {results_path}")

        # Save summary report
        self.save_summary(coercion_results)

        logging.info(f"All results saved to {self.output_dir}")

        return
    
    def save_detailed_results(self, coercion_results):
        """
        Save detailed coercion detection results to a CSV file.
        
        Args:
            coercion_results (pd.DataFrame): DataFrame containing detailed detection results
            
        Returns:
            str: Path to the saved CSV file
        """
        csv_path = os.path.join(self.output_dir, "detailed_results.csv")
        coercion_results.to_csv(csv_path, sep=";", index=False)
        logging.info(f"Saved detailed results to {csv_path}")
        return csv_path
    