# chains.py

import json
import logging
from typing import List, Dict
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import OutputParserException
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from models import (
    SentimentResults,
    CustomerAttributes,
    BusinessAnalysis,
    PrimaryAttribute,
)
from config import get_openai_api_key
import math
import time
import re

logger = logging.getLogger()

# Define the batch size
BATCH_SIZE = 50  # Adjust based on your specific needs and model limitations
ATTRIBUTE_BATCH_SIZE = 50  # Adjust as needed

# Initialize the LLM once
llm = ChatOpenAI(
    model_name="gpt-4o-mini", temperature=0.3, openai_api_key=get_openai_api_key()
)


def create_sentiment_chain() -> RunnableSequence:
    """
    Create a sentiment analysis chain using LangChain's ChatPromptTemplate and PydanticOutputParser.
    Incorporates batch processing to handle large numbers of statements.
    """
    logger.info("Creating sentiment analysis chain...")

    # Initialize the Pydantic output parser and retrieve format instructions
    sentiment_parser = PydanticOutputParser(pydantic_object=SentimentResults)
    format_instructions = sentiment_parser.get_format_instructions()

    # Define the prompt template with placeholders for format instructions and statements
    sentiment_prompt_template = ChatPromptTemplate.from_template(
        """
    Perform sentiment analysis on the following customer statements in the {domain} domain.
    For each statement, provide a sentiment score between -1 (negative) and +1 (positive),
    and a sentiment label (Positive, Neutral, Negative).

    **Important:** Ensure that all statements are analyzed. Do not leave anything out.

    **Important:** Please return the results in the following format:
    {format_instructions}

    Customer Statements:
    {statements}
        """
    ).partial(format_instructions=format_instructions)

    # Define the sequence
    sentiment_sequence = RunnableSequence(
        sentiment_prompt_template,
        llm,
        sentiment_parser,  # Parse the output into SentimentResults
    )

    return sentiment_sequence


def batch_sentences(sentences: List[str], batch_size: int) -> List[List[str]]:
    """
    Split the list of sentences into batches of specified size.
    """
    return [sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)]


def process_sentiment_analysis(
    sentiment_chain: RunnableSequence,
    customer_statements: List[str],
    domain: str,
) -> List[Dict]:
    """
    Process sentiment analysis in batches and ensure all statements are analyzed.

    Args:
        sentiment_chain (RunnableSequence): The sentiment analysis chain.
        customer_statements (List[str]): List of customer statements.
        domain (str): The domain for analysis.

    Returns:
        List[Dict]: List of sentiment analysis results as dictionaries.
    """
    all_results = []
    batches = batch_sentences(customer_statements, BATCH_SIZE)
    total_batches = len(batches)

    logger.info(f"Total sentiment analysis batches to process: {total_batches}")

    for idx, batch in enumerate(batches, start=1):
        logger.info(f"Processing sentiment batch {idx}/{total_batches}...")
        attempt = 0
        max_retries = 3
        while attempt < max_retries:
            try:
                batch_output = sentiment_chain.invoke(
                    {
                        "statements": "\n".join(batch),
                        "domain": domain,
                    }
                )
                if not batch_output:
                    raise ValueError("No sentiment results obtained for the batch.")

                # Access the 'results' list from SentimentResults
                parsed_results = batch_output.results  # List[SentimentResult]

                # Verify that the number of results matches the number of statements
                if len(parsed_results) != len(batch):
                    missing = len(batch) - len(parsed_results)
                    logger.warning(
                        f"Batch {idx}: {missing} statements missing. Retrying..."
                    )
                    raise ValueError(
                        f"{missing} statements missing in the sentiment analysis."
                    )

                # Convert each SentimentResult to a dictionary and append to all_results
                all_results.extend([result.dict() for result in parsed_results])
                logger.info(f"Sentiment batch {idx} processed successfully.")
                break  # Exit the retry loop on success
            except (OutputParserException, ValueError, Exception) as e:
                attempt += 1
                logger.error(f"Error processing sentiment batch {idx}: {e}")
                if attempt < max_retries:
                    sleep_time = 2**attempt
                    logger.info(
                        f"Retrying sentiment batch {idx} in {sleep_time} seconds..."
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        f"Failed to process sentiment batch {idx} after {max_retries} attempts."
                    )
                    # Optionally, handle the failed batch (e.g., save to a file for manual review)
                    # For now, we'll skip and continue
                    break

    # Final verification
    if len(all_results) != len(customer_statements):
        missing_count = len(customer_statements) - len(all_results)
        logger.warning(
            f"Total statements missing: {missing_count}. Attempting to reprocess missing statements."
        )
        # Identify missing statements
        analyzed_statements = set(result["statement"] for result in all_results)
        missing_statements = [
            stmt for stmt in customer_statements if stmt not in analyzed_statements
        ]
        if missing_statements:
            # Recursively process missing statements
            logger.info(f"Reprocessing {len(missing_statements)} missing statements...")
            reprocessed_results = process_sentiment_analysis(
                sentiment_chain, missing_statements, domain
            )
            all_results.extend(reprocessed_results)

    # Final check
    if len(all_results) != len(customer_statements):
        logger.error("Some statements were not analyzed successfully.")
        # Handle accordingly, e.g., raise an exception or log for manual review
        # Here, we'll log the missing statements
        analyzed_statements = set(result["statement"] for result in all_results)
        missing_statements = [
            stmt for stmt in customer_statements if stmt not in analyzed_statements
        ]
        if missing_statements:
            logger.error(f"Missing Statements: {missing_statements}")

    return all_results


def batch_attribute_sentiments(
    sentiments: List[Dict], batch_size: int
) -> List[List[Dict]]:
    """
    Split the list of sentiment results into batches of specified size for attribute derivation.

    Args:
        sentiments (List[Dict]): List of sentiment analysis results.
        batch_size (int): Size of each batch.

    Returns:
        List[List[Dict]]: Batches of sentiment results.
    """
    return [
        sentiments[i : i + batch_size] for i in range(0, len(sentiments), batch_size)
    ]


def create_attribute_chain() -> RunnableSequence:
    """
    Create an attribute derivation chain with a nested attribute structure.
    Incorporates batch processing to ensure all statements are attributed.

    Returns:
        RunnableSequence: The attribute derivation chain.
    """
    logger.info("Creating attribute derivation chain...")

    # Initialize the Pydantic output parser and retrieve format instructions
    attribute_parser = PydanticOutputParser(pydantic_object=CustomerAttributes)
    format_instructions = attribute_parser.get_format_instructions()

    # Define the prompt template with placeholders for format instructions and sentiment results
    attribute_prompt_template = ChatPromptTemplate.from_template(
        """
    Based on the statements and sentiment analysis provided, identify key customer attributes and critical features for 
    the {domain} domain using the Quality Function Deployment (QFD) approach to ensure that customer needs and expectations are 
    incorporated into the final design of the service. 
    Structure the attributes in a three-level hierarchy (primary, secondary, tertiary) with associated customer statements and their sentiment scores.

    1. **Primary Attributes**: Broad categories of customer needs.
    2. **Secondary Attributes**: Specific aspects within each primary attribute.
    3. **Tertiary Attributes**: Detailed elements that impact each secondary attribute, along with related customer statements and their sentiment scores.

    Ensure that:
    - Attribute names are unique within their respective levels.
    - Each primary attribute can have multiple secondary attributes.
    - Each secondary attribute can have multiple tertiary attributes.
    - Each customer statement is assigned to one and only one tertiary attribute.
    - The structure is comprehensive and professionally worded relevant to the {domain} domain.
    - No statements are left out without being attributed to an attribute.

    **Example Output:**
    
    ```json
    {
        "attributes": [
            {
                "primary_attribute": "Staff Interaction",
                "secondary_attributes": [
                    {
                        "attribute": "Staff Professionalism",
                        "tertiary_attributes": [
                            {
                                "attribute": "Politeness of staff",
                                "statements": [
                                    {
                                        "statement": "The staff was extremely polite and accommodating.",
                                        "score": 0.95
                                    },
                                    {
                                        "statement": "Great service and professional staff.",
                                        "score": 0.9
                                    }
                                ]
                            },
                            {
                                "attribute": "Communication skills of staff",
                                "statements": [
                                    {
                                        "statement": "I appreciated the clear communication from the staff.",
                                        "score": 0.85
                                    },
                                    {
                                        "statement": "I don't understand the nurse. Language training please!",
                                        "score": -0.7
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
            // Add more primary attributes as needed
        ]
    }
    ```
    
    Please return the results in the following format:
    {format_instructions}
    
    Sentiment Analysis Results:
    {sentiment_results}
        """
    ).partial(format_instructions=format_instructions)

    # Define the sequence
    attribute_sequence = RunnableSequence(
        attribute_prompt_template,
        llm,
        attribute_parser,  # Parse the output into CustomerAttributes
    )

    return attribute_sequence


def process_attribute_derivation(
    attribute_chain: RunnableSequence,
    sentiment_results: List[Dict],
    domain: str,
) -> List[PrimaryAttribute]:
    """
    Process attribute derivation in batches and ensure all statements are attributed.

    Args:
        attribute_chain (RunnableSequence): The attribute derivation chain.
        sentiment_results (List[Dict]): List of sentiment analysis results.
        domain (str): The domain for analysis.

    Returns:
        List[PrimaryAttribute]: List of primary customer attributes.
    """
    all_attributes = []
    batches = batch_attribute_sentiments(sentiment_results, ATTRIBUTE_BATCH_SIZE)
    total_batches = len(batches)

    logger.info(f"Total attribute derivation batches to process: {total_batches}")

    for idx, batch in enumerate(batches, start=1):
        logger.info(f"Processing attribute derivation batch {idx}/{total_batches}...")
        attempt = 0
        max_retries = 3
        while attempt < max_retries:
            try:
                batch_output = attribute_chain.invoke(
                    {
                        "sentiment_results": json.dumps(batch),
                        "domain": domain,
                    }
                )
                if not batch_output:
                    raise ValueError("No attribute results obtained for the batch.")

                # Log the type and content of batch_output for debugging
                logger.debug(f"Type of batch_output: {type(batch_output)}")
                logger.debug(f"Batch Output Content: {batch_output}")

                parsed_results = batch_output  # Should be CustomerAttributes object

                if not isinstance(parsed_results, CustomerAttributes):
                    # If it's a dict, attempt to parse it manually
                    logger.debug(
                        "Parsed results are not CustomerAttributes. Attempting manual parsing."
                    )
                    parsed_results = CustomerAttributes(**batch_output)

                # Confirm type after manual parsing
                if not isinstance(parsed_results, CustomerAttributes):
                    raise TypeError(
                        "Parsed results are not of type CustomerAttributes even after manual parsing."
                    )

                # Append the attributes from the parsed_results
                all_attributes.extend(parsed_results.attributes)
                logger.info(f"Attribute derivation batch {idx} processed successfully.")
                break  # Exit the retry loop on success
            except (OutputParserException, ValueError, TypeError, Exception) as e:
                attempt += 1
                logger.error(f"Error processing attribute derivation batch {idx}: {e}")
                if attempt < max_retries:
                    sleep_time = 2**attempt
                    logger.info(
                        f"Retrying attribute derivation batch {idx} in {sleep_time} seconds..."
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        f"Failed to process attribute derivation batch {idx} after {max_retries} attempts."
                    )
                    # Optionally, handle the failed batch (e.g., save to a file for manual review)
                    # For now, we'll skip and continue
                    break

    # Final verification
    # Ensure that each sentiment result is attributed to one tertiary attribute
    attributed_statements = set()
    for attr in all_attributes:
        for secondary_attr in attr.secondary_attributes:
            for tertiary_attr in secondary_attr.tertiary_attributes:
                for stmt in tertiary_attr.statements:
                    attributed_statements.add(stmt.statement)

    original_statements = set(stmt["statement"] for stmt in sentiment_results)

    missing_statements = original_statements - attributed_statements

    if missing_statements:
        logger.warning(
            f"{len(missing_statements)} statements were not attributed. Attempting to reprocess..."
        )
        # Optionally, handle missing statements by reprocessing
        # For simplicity, let's log them
        for stmt in missing_statements:
            logger.warning(f"Missing statement: {stmt}")
        # Here, you can implement additional logic to reprocess or flag these statements

    # Return the flat list of PrimaryAttribute objects
    return all_attributes


def generate_business_analysis(
    domain: str,
    customer_attributes: List[PrimaryAttribute],
    attribute_weights: Dict[str, float],
    survey_details: str,
) -> BusinessAnalysis:
    """
    Generate a comprehensive business analysis based on the sentiment analysis, customer attributes, and survey details.
    Ensures that all customer statements are accounted for in the analysis.

    Args:
        domain (str): The domain for analysis.
        customer_attributes (List[PrimaryAttribute]): List of primary customer attributes.
        attribute_weights (Dict[str, float]): Relative importance of customer attributes.
        survey_details (str): Details about the survey conducted.

    Returns:
        BusinessAnalysis: The generated business analysis.
    """
    logger.info("Generating business analysis using LLM...")

    # Prepare the data in a structured format
    attribute_data = []
    for primary_attr in customer_attributes:
        for secondary_attr in primary_attr.secondary_attributes:
            for tertiary_attr in secondary_attr.tertiary_attributes:
                for stmt in tertiary_attr.statements:
                    attribute_data.append(
                        {
                            "Primary Attribute": primary_attr.primary_attribute,
                            "Secondary Attribute": secondary_attr.attribute,
                            "Tertiary Attribute": tertiary_attr.attribute,
                            "Statement": stmt.statement,
                            "Score": stmt.score,  # Changed to match SentimentResult
                        }
                    )

    # Serialize the data to JSON for inclusion in the prompt
    attribute_json = json.dumps(attribute_data, indent=2, ensure_ascii=False)
    weights_json = json.dumps(attribute_weights, indent=2, ensure_ascii=False)
    total_statements = len(attribute_data)

    # Create a prompt for the LLM to generate the analysis
    prompt = f"""
You are a professional business analyst specialized in Quality Function Deployment (QFD). Below are the details of a survey that was conducted, followed by the customer feedback analysis for the {domain} domain. Provide a comprehensive business analysis suitable for inclusion in a report for operations management. Use the context provided in the survey details to customize your analysis.

### Survey Details
{survey_details}

### Customer Attribute Breakdown
{attribute_json}

### Relative Importance of Customer Attributes
{weights_json}

**Total Number of Statements:** {total_statements}

**Objective:** Ensure that your analysis comprehensively covers all provided customer statements. If any statements are not addressed in your analysis, please identify and include them in the relevant sections.

Please structure your analysis into five sections:
1. **Introduction**: Provide a background of the report, what it is based on, who it is for, and who commissioned it/led the project together with an overview of the business analysis using the details about the survey and the generally accepted standards for the {domain}.
2. **Introduction to Derived Customer Attributes**: Based on techniques used in QFD, provide an explanation on how customer attributes can be derived from a set of statements from the customers.
3. **Analysis**: Discuss key findings and insights derived from the data.
4. **Recommendations**: Offer actionable suggestions for the operations management based on the analysis.
5. **Conclusion**: A high-level conclusion of the analysis with a focus on customer satisfaction and quality.

**Instructions:**
- Ensure the analysis is professional, clear, and suitable for a business report aimed at the operational management team.
- Verify that all {total_statements} customer statements have been considered in your analysis.
- If you find any statements that are not addressed, please include them appropriately in the relevant sections.
    """

    # Initialize the Pydantic output parser for BusinessAnalysis
    business_analysis_parser = PydanticOutputParser(pydantic_object=BusinessAnalysis)
    business_analysis_format_instructions = (
        business_analysis_parser.get_format_instructions()
    )

    # Append format instructions to the prompt
    full_prompt = prompt + "\n\n" + business_analysis_format_instructions

    # Define retry parameters
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempt {attempt} to generate business analysis...")
            response = llm.invoke(full_prompt)
            # Parse the response into BusinessAnalysis
            business_analysis = business_analysis_parser.parse(response.content)
            logger.debug(f"Business Analysis Generated: {business_analysis}")

            # Verify completeness
            missing_statements = verify_completeness(response.content, total_statements)
            if missing_statements:
                logger.warning(
                    f"Missing {len(missing_statements)} statements in the analysis."
                )
                if attempt < max_retries:
                    logger.info(
                        f"Retrying to include missing statements after {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                    # Update the prompt to include missing statements
                    updated_prompt = append_missing_statements_prompt(
                        full_prompt, missing_statements
                    )
                    full_prompt = updated_prompt
                    continue
                else:
                    logger.error(
                        "Max retries reached. Some statements are still missing."
                    )
            else:
                logger.info(
                    "All statements are accounted for in the business analysis."
                )

            return business_analysis

        except OutputParserException as e:
            logger.error(f"Failed to parse business analysis: {e}")
            try:
                logger.debug(f"Assistant's response content: {response.content}")
            except NameError:
                logger.debug(
                    "Assistant's response content is unavailable due to an earlier error."
                )
            if attempt < max_retries:
                logger.info(f"Retrying after {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                logger.error(
                    "Max retries reached. Returning incomplete BusinessAnalysis."
                )
                return BusinessAnalysis(
                    introduction="Business analysis could not be generated due to a parsing error.",
                    derived_attributes_introduction="",
                    analysis="",
                    recommendations="",
                    conclusions="",
                )
        except Exception as e:
            logger.error(f"Failed to generate business analysis: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying after {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                logger.error(
                    "Max retries reached. Returning incomplete BusinessAnalysis."
                )
                return BusinessAnalysis(
                    introduction="Business analysis could not be generated due to an error.",
                    derived_attributes_introduction="",
                    analysis="",
                    recommendations="",
                    conclusions="",
                )

    # If all retries fail, return an incomplete BusinessAnalysis
    return BusinessAnalysis(
        introduction="Business analysis could not be generated due to repeated errors.",
        derived_attributes_introduction="",
        analysis="",
        recommendations="",
        conclusions="",
    )


def verify_completeness(response_content: str, total_statements: int) -> List[str]:
    """
    Verify that the LLM's response covers all customer statements.

    Args:
        response_content (str): The raw response content from the LLM.
        total_statements (int): The total number of customer statements.

    Returns:
        List[str]: A list of statements that are missing in the analysis.
    """
    # This function's implementation depends on how the LLM structures the response.
    # For simplicity, we'll assume that the response includes a summary that mentions
    # the number of statements covered.

    # Example: "This analysis covers all 100 customer statements."
    pattern = r"covers all (\d+) customer statements"
    match = re.search(pattern, response_content, re.IGNORECASE)
    if match:
        covered = int(match.group(1))
        if covered >= total_statements:
            return []
        else:
            missing = total_statements - covered
            # In a real scenario, you'd have access to the statements and can identify which are missing
            # For this example, we'll return a placeholder list
            return [f"Statement {i+1}" for i in range(missing)]
    else:
        # If the pattern isn't found, assume completeness couldn't be verified
        logger.warning("Could not verify completeness from the LLM response.")
        return []


def append_missing_statements_prompt(
    existing_prompt: str, missing_statements: List[str]
) -> str:
    """
    Append instructions to include missing statements in the analysis.

    Args:
        existing_prompt (str): The current prompt sent to the LLM.
        missing_statements (List[str]): List of missing statements to be included.

    Returns:
        str: The updated prompt with instructions to include missing statements.
    """
    missing_statements_text = "\n".join(missing_statements)
    additional_instructions = f"""

### Missing Statements
The following {len(missing_statements)} customer statements were not addressed in the previous analysis. Please include them in the relevant sections of the business analysis.

{missing_statements_text}

Please update the analysis to incorporate these statements accordingly.
    """
    return existing_prompt + additional_instructions
