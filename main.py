# main.py

import logging
import argparse
import json

from config import configure_logging
from data_loading import load_customer_statements, load_survey_details
from chains import (
    create_sentiment_chain,
    create_attribute_chain,
    process_sentiment_analysis,
    process_attribute_derivation,
    generate_business_analysis,
)
from analysis import calculate_relative_importance
from visualization import generate_visualizations
from report import generate_final_report, save_attributes_json
from langchain_core.exceptions import OutputParserException

from models import (
    BusinessAnalysis,
    PrimaryAttribute,
)


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Customer Feedback Analysis with Nested Attributes"
    )
    parser.add_argument(
        "--domain", type=str, default="Healthcare Services", help="Domain for analysis"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./survey.txt",
        help="Path to customer statements file",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_arguments()
    domain = args.domain
    input_file = args.input_file
    log_level = args.log_level.upper()

    # Configure logging
    logger = configure_logging(log_level)

    logger.info("Starting the customer attribute analysis pipeline...")

    # Load customer statements
    customer_statements = load_customer_statements(input_file)

    if not customer_statements:
        logger.error("No customer statements loaded. Exiting.")
        return

    # Create LLM chain for sentiment analysis
    sentiment_chain = create_sentiment_chain()
    # Remove the line that creates attribute_chain, as it is now created within process_attribute_derivation
    # attribute_chain = create_attribute_chain()  # Remove or comment out this line

    # Step 1: Run sentiment analysis with batch processing
    logger.info("Performing sentiment analysis with batch processing...")
    try:
        all_sentiment_results = process_sentiment_analysis(
            sentiment_chain, customer_statements, domain
        )
        logger.debug(f"Total Sentiment Results: {len(all_sentiment_results)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during sentiment analysis: {e}")
        return

    # Count neutral statements
    neutral_count = sum(
        1 for result in all_sentiment_results if result["label"] == "Neutral"
    )
    logger.info(f"Total Neutral Statements: {neutral_count}")
    if neutral_count == 0:
        logger.warning(
            "No neutral statements were detected. Review sentiment analysis parameters."
        )

    # Step 2: Run attribute derivation chain with batch processing
    logger.info("Deriving customer attributes with batch processing...")
    try:
        # Pass the create_attribute_chain function as a callable
        parsed_customer_attributes = process_attribute_derivation(
            create_attribute_chain,  # Pass the function, not an instance
            all_sentiment_results,  # Already a list of dicts
            domain,
        )

        if not parsed_customer_attributes:
            logger.error("No customer attributes derived. Exiting.")
            return

        logger.debug(f"Parsed Customer Attributes: {parsed_customer_attributes}")

        # Save attributes.json for manual review
        save_attributes_json(parsed_customer_attributes)

    except OutputParserException as e:
        logger.error(f"Failed to parse customer attributes: {e}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred during attribute derivation: {e}")
        return

    # Step 3: Calculate relative importance
    attribute_weights = calculate_relative_importance(parsed_customer_attributes)
    logger.debug(f"Attribute Weights before Conversion: {attribute_weights}")

    if not attribute_weights:
        logger.error("No attribute weights calculated. Exiting.")
        return

    # Step 4: Generate visualizations
    generate_visualizations(attribute_weights)

    # Step 5: Generate business analysis
    survey_details = load_survey_details("./survey_details.txt")
    business_analysis = generate_business_analysis(
        domain, parsed_customer_attributes, attribute_weights, survey_details
    )

    # Step 6: Generate the comprehensive report
    generate_final_report(
        domain, parsed_customer_attributes, attribute_weights, business_analysis
    )

    logger.info("Customer attribute analysis pipeline completed successfully.")


if __name__ == "__main__":
    main()
