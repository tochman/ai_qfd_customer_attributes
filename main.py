import argparse

from config import configure_logging
from data_loading import load_customer_statements, load_survey_details
from chains import (
    create_relevance_chain,
    process_relevance_classification,
    filter_relevant_statements,
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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Customer Feedback Analysis with Nested Attributes"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="Healthcare Services",
        help="Domain for analysis",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./survey.txt",
        help="Path to customer statements file",
    )
    parser.add_argument(
        "--survey_details_file",
        type=str,
        default="./survey_details.txt",
        help="Path to the survey details file",
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
    args = parse_arguments()
    domain = args.domain
    input_file = args.input_file
    survey_details_file = args.survey_details_file
    log_level = args.log_level.upper()

    logger = configure_logging(log_level)

    logger.info("Starting the customer attribute analysis pipeline...")

    customer_statements = load_customer_statements(input_file)

    if not customer_statements:
        logger.error("No customer statements loaded. Exiting.")
        return

    # Relevance Classification
    logger.info("Classifying statements based on relevance...")
    relevance_chain = create_relevance_chain()
    try:
        relevance_results = process_relevance_classification(
            relevance_chain, customer_statements, domain
        )
        customer_statements = filter_relevant_statements(relevance_results)
        logger.info(f"Total relevant statements: {len(customer_statements)}")
    except Exception as e:
        logger.error(f"An error occurred during relevance classification: {e}")
        return

    if not customer_statements:
        logger.error("No relevant customer statements found. Exiting.")
        return

    # Create LLM chain for sentiment analysis
    sentiment_chain = create_sentiment_chain()

    # Run sentiment analysis with batch processing
    logger.info("Performing sentiment analysis with batch processing...")
    try:
        all_sentiment_results = process_sentiment_analysis(
            sentiment_chain, customer_statements, domain
        )
        logger.debug(f"Total Sentiment Results: {len(all_sentiment_results)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during sentiment analysis: {e}")
        return

    neutral_count = sum(
        1 for result in all_sentiment_results if result["label"] == "Neutral"
    )
    logger.info(f"Total Neutral Statements: {neutral_count}")
    if neutral_count == 0:
        logger.warning(
            "No neutral statements were detected. Review sentiment analysis parameters."
        )

    # Run attribute derivation chain with batch processing
    logger.info("Deriving customer attributes with batch processing...")
    try:
        parsed_customer_attributes = process_attribute_derivation(
            create_attribute_chain,
            all_sentiment_results,
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

    # Calculate relative importance
    attribute_weights = calculate_relative_importance(parsed_customer_attributes)
    logger.debug(f"Attribute Weights before Conversion: {attribute_weights}")

    if not attribute_weights:
        logger.error("No attribute weights calculated. Exiting.")
        return

    # Generate visualizations
    generate_visualizations(attribute_weights)

    # Generate business analysis
    survey_details = load_survey_details(survey_details_file)
    business_analysis = generate_business_analysis(
        domain, parsed_customer_attributes, attribute_weights, survey_details
    )

    # Generate the comprehensive report
    generate_final_report(
        domain, parsed_customer_attributes, attribute_weights, business_analysis
    )

    logger.info("Customer attribute analysis pipeline completed successfully.")


if __name__ == "__main__":
    main()
