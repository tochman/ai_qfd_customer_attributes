# analysis.py

from typing import List, Dict
from models import PrimaryAttribute
import logging

logger = logging.getLogger()


def calculate_relative_importance(
    customer_attributes: List[PrimaryAttribute]
) -> Dict[str, float]:
    """
    Calculate the relative importance of customer attributes based on sentiment scores.
    """
    logger.info("Calculating relative importance of customer attributes...")
    attribute_weights = {}
    total_score = 0.0

    # Initialize weights for primary attributes
    for attr in customer_attributes:
        primary_attr = attr.primary_attribute
        attribute_weights[primary_attr] = 0.0

    # Aggregate absolute sentiment scores to avoid negative weights
    for primary_attr in customer_attributes:
        for secondary_attr in primary_attr.secondary_attributes:
            for tertiary_attr in secondary_attr.tertiary_attributes:
                for stmt in tertiary_attr.statements:
                    score = abs(
                        stmt.score
                    )  # Take absolute value of the score
                    attribute_weights[primary_attr.primary_attribute] += score
                    total_score += score
                    logger.debug(
                        f"Adding absolute score {score:.2f} from statement '{stmt.statement}' to attribute '{primary_attr.primary_attribute}'"
                    )

    if total_score == 0:
        logger.error(
            "Total sentiment score is zero, cannot calculate attribute weights."
        )
        return attribute_weights

    # Convert to percentages and normalize so that the sum is 100%
    for attr in attribute_weights:
        attribute_weights[attr] = round(
            (attribute_weights[attr] / total_score) * 100, 2
        )

    # Calculate total to verify correctness
    total_weight = sum(attribute_weights.values())
    logger.info(f"Total weight calculated (should be ~100%): {total_weight}%")

    return attribute_weights
