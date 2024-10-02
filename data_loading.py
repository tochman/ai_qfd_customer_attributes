import logging
from typing import List

logger = logging.getLogger()


def load_customer_statements(file_path: str = "./survey.txt") -> List[str]:
    """
    Load customer statements from a text file.
    """
    try:
        logger.info(f"Loading customer statements from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as file:
            statements = [line.strip() for line in file if line.strip()]
        logger.debug(f"Loaded {len(statements)} customer statements.")
        return statements
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"An error occurred while loading customer statements: {e}")
        return []


def load_survey_details(file_path: str = "./survey_details.txt") -> str:
    """
    Load survey details from a text file.
    """
    try:
        logger.info(f"Loading survey details from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as file:
            survey_details = file.read().strip()
        logger.debug(f"Loaded survey details: {survey_details}")
        return survey_details
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return ""
    except Exception as e:
        logger.error(f"An error occurred while loading survey details: {e}")
        return ""
