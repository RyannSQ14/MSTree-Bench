import logging
import os
import sys
import platform
from datetime import datetime
import multiprocessing
import numpy as np

def get_system_info():
    """Get information about the system."""
    info = {}
    
    # Get OS information
    info["os"] = platform.system()
    info["os_version"] = platform.version()
    info["os_release"] = platform.release()
    
    # Get CPU information
    if platform.system() == "Darwin":  # macOS
        try:
            import subprocess
            # Get CPU model
            cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
            cpu_model = subprocess.check_output(cmd).decode("utf-8").strip()
            info["cpu_model"] = cpu_model
            
            # Get CPU clock speed
            cmd = ["sysctl", "-n", "hw.cpufrequency"]
            try:
                cpu_freq = subprocess.check_output(cmd).decode("utf-8").strip()
                info["cpu_clock"] = f"{int(cpu_freq)/1000000000:.2f} GHz"
            except:
                info["cpu_clock"] = "Unknown"
        except:
            info["cpu_model"] = "Unknown"
            info["cpu_clock"] = "Unknown"
    elif platform.system() == "Linux":
        try:
            import subprocess
            # Try to get CPU model
            cmd = ["cat", "/proc/cpuinfo"]
            proc_info = subprocess.check_output(cmd).decode("utf-8").strip()
            for line in proc_info.split("\n"):
                if "model name" in line:
                    info["cpu_model"] = line.split(":")[1].strip()
                    break
            else:
                info["cpu_model"] = "Unknown"
                
            # Try to get CPU clock speed
            for line in proc_info.split("\n"):
                if "cpu MHz" in line:
                    freq = float(line.split(":")[1].strip())
                    info["cpu_clock"] = f"{freq/1000:.2f} GHz"
                    break
            else:
                info["cpu_clock"] = "Unknown"
        except:
            info["cpu_model"] = "Unknown" 
            info["cpu_clock"] = "Unknown"
    elif platform.system() == "Windows":
        try:
            import subprocess
            cmd = ["wmic", "cpu", "get", "name"]
            cpu_model = subprocess.check_output(cmd).decode("utf-8").strip().split("\n")[1]
            info["cpu_model"] = cpu_model
            
            cmd = ["wmic", "cpu", "get", "maxclockspeed"]
            cpu_freq = subprocess.check_output(cmd).decode("utf-8").strip().split("\n")[1]
            info["cpu_clock"] = f"{int(cpu_freq)/1000:.2f} GHz"
        except:
            info["cpu_model"] = platform.processor()
            info["cpu_clock"] = "Unknown"
    else:
        info["cpu_model"] = platform.processor()
        info["cpu_clock"] = "Unknown"
    
    # Get CPU cores
    info["cpu_cores"] = multiprocessing.cpu_count()
    
    # Get Python version
    info["python_version"] = sys.version.split()[0]
    
    # Get random number generator info
    info["random_generator"] = f"Python's random module (Mersenne Twister algorithm)"
    info["numpy_random"] = f"NumPy {np.__version__} (PCG64 algorithm)"
    
    return info

def format_system_info(info):
    """Format system information for logging."""
    lines = [
        "=" * 80,
        "SYSTEM INFORMATION",
        "=" * 80,
        f"Operating System: {info['os']} {info['os_release']} ({info['os_version']})",
        f"CPU: {info['cpu_model']}",
        f"CPU Clock: {info['cpu_clock']}",
        f"CPU Cores: {info['cpu_cores']}",
        f"Python Version: {info['python_version']}",
        f"Programming Language: Python",
        f"Random Number Generator: {info['random_generator']}",
        f"NumPy Random Generator: {info['numpy_random']}",
        "=" * 80
    ]
    return "\n".join(lines)

def setup_logger():
    """Set up and return a logger for logging all operations."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger('mst_logger')
    logger.setLevel(logging.DEBUG)
    
    # Create a file handler
    log_filename = f"logs/execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    
    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    
    # Add formatters to handlers
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to file: {log_filename}")
    
    # Log system information
    system_info = get_system_info()
    logger.info(format_system_info(system_info))
    
    return logger

if __name__ == "__main__":
    # Test the logger
    logger = setup_logger()
    logger.info("Test log message")
    logger.debug("Test debug message")
    logger.warning("Test warning message")
    logger.error("Test error message") 