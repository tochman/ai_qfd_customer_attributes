# config.py

import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
    
def configure_logging(log_level="INFO"):
    """
    Configure logging with both console and file handlers.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels for file

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler for INFO level and above
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level, logging.INFO))
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler for DEBUG level
    fh = logging.FileHandler("debug.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

  

    return logger


def get_openai_api_key():
    """
    Retrieve the OpenAI API key from environment variables.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logging.getLogger().error(
            "OpenAI API key not found. Please set it in your .env file."
        )
        raise ValueError("OpenAI API key not found. Please set it in your .env file.")
    return openai_api_key
