import os
import sys
import time
from experiment import run_all_experiments
from report_generator import generate_report
from logger import setup_logger

def main():
    """
    Main entry point for the MST assignment.
    Runs experiments and generates reports.
    """
    # Set up logging
    logger = setup_logger()
    
    logger.info("=" * 80)
    logger.info("Minimum Spanning Tree Algorithms - Experimental Analysis")
    logger.info("=" * 80)
    
    # Create necessary directories
    os.makedirs("reports", exist_ok=True)
    
    # Run all experiments
    logger.info("\nRunning experiments (this may take a while)...")
    start_time = time.time()
    
    try:
        # Pass the logger to run_all_experiments
        run_all_experiments(logger)
        
        end_time = time.time()
        logger.info(f"\nAll experiments completed in {end_time - start_time:.2f} seconds.")
        
        # Generate comprehensive report
        logger.info("\nGenerating final report...")
        generate_report(logger)
        
        logger.info("\nAssignment completed successfully!")
        logger.info(f"Report available at: {os.path.abspath('reports/mst_analysis_report.html')}")
        logger.info("\nYou can view the report by opening the HTML file in your browser.")
    
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 