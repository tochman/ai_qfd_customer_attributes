import json
import logging
from typing import List, Dict, Callable
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
    SecondaryAttribute
)
from config import get_openai_api_key
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
    and a sentiment label (Positive, Neutral, Negative) based on the following thresholds:
    - Negative: -1.0 to -0.3
    - Neutral: -0.3 to 0.3
    - Positive: 0.3 to 1.0

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


def create_attribute_chain(
    existing_attributes_json: str, domain: str
) -> RunnableSequence:
    """
    Create an attribute derivation chain with a nested attribute structure.
    Incorporates batch processing and global context to ensure consistency across batches.

    Args:
        existing_attributes_json (str): JSON string of the attributes identified so far.
        domain (str): The domain for analysis.

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
Based on the statements and sentiment analysis provided, assign each statement to the most appropriate existing customer attribute or create a new one if necessary. Use the Quality Function Deployment (QFD) approach to ensure that customer needs and expectations are incorporated into the final design of the service. Structure the attributes in a three-level hierarchy (primary, secondary, tertiary) with associated customer statements and their sentiment scores.

**Existing Attributes:**
{existing_attributes}

**Instructions:**
- Use the existing attributes where appropriate.
- Only create new attributes if a statement does not fit into any existing attribute.
- Ensure attribute names are consistent and unique within their respective levels.
- Each customer statement must be assigned to one and only one tertiary attribute.
- The structure should be comprehensive and professionally worded relevant to the {domain} domain.
- Do not leave any statements unassigned.

**Example Output:**
(Your escaped JSON example here)

Please return the results in the following format:
{format_instructions}

Sentiment Analysis Results:
{sentiment_results}
        """
    ).partial(
        format_instructions=format_instructions,
        existing_attributes=existing_attributes_json,
        domain=domain,
    )

    # Define the sequence
    attribute_sequence = RunnableSequence(
        attribute_prompt_template,
        llm,
        attribute_parser,  # Parse the output into CustomerAttributes
    )

    return attribute_sequence


def process_attribute_derivation(
    create_attribute_chain: Callable[..., RunnableSequence],
    sentiment_results: List[Dict],
    domain: str,
) -> List[PrimaryAttribute]:
    """
    Process attribute derivation in batches and ensure all statements are attributed.
    Maintains a global attribute list for consistency across batches.

    Args:
        create_attribute_chain (Callable[..., RunnableSequence]): Function to create the attribute derivation chain.
        sentiment_results (List[Dict]): List of sentiment analysis results.
        domain (str): The domain for analysis.

    Returns:
        List[PrimaryAttribute]: List of primary customer attributes.
    """
    global_attributes = []  # Global attribute list
    batches = batch_attribute_sentiments(sentiment_results, ATTRIBUTE_BATCH_SIZE)
    total_batches = len(batches)

    logger.info(f"Total attribute derivation batches to process: {total_batches}")

    for idx, batch in enumerate(batches, start=1):
        logger.info(f"Processing attribute derivation batch {idx}/{total_batches}...")
        attempt = 0
        max_retries = 3
        while attempt < max_retries:
            try:
                # Serialize global attributes to JSON
                existing_attributes_json = json.dumps(
                    [attr.dict() for attr in global_attributes],
                    indent=2,
                    ensure_ascii=False,
                )

                # Create the attribute chain with the existing attributes and domain
                chain = create_attribute_chain(existing_attributes_json, domain)

                batch_output = chain.invoke({"sentiment_results": json.dumps(batch)})

                if not batch_output:
                    raise ValueError("No attribute results obtained for the batch.")

                parsed_results = batch_output  # Should be CustomerAttributes object

                if not isinstance(parsed_results, CustomerAttributes):
                    parsed_results = CustomerAttributes(**batch_output)

                # Extract attributes from the batch
                batch_attributes = parsed_results.attributes

                # Merge batch attributes with global attributes
                global_attributes = merge_attributes(
                    global_attributes + batch_attributes
                )

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
                    break

    # Final verification and any additional processing
    logger.info("Performing final verification of attributed statements.")

    # Collect all attributed statements
    attributed_statements = set()
    for primary_attr in global_attributes:
        for secondary_attr in primary_attr.secondary_attributes:
            for tertiary_attr in secondary_attr.tertiary_attributes:
                for stmt in tertiary_attr.statements:
                    attributed_statements.add(stmt.statement.strip())

    # Collect all original statements
    original_statements = set(item["statement"].strip() for item in sentiment_results)

    # Identify unassigned statements
    unassigned_statements = original_statements - attributed_statements

    if unassigned_statements:
        logger.warning(
            f"{len(unassigned_statements)} statements were not attributed. Assigning them to 'Uncategorized'."
        )
        logger.debug(f"Unassigned Statements: {unassigned_statements}")

        # Create an 'Uncategorized' attribute if it doesn't exist
        uncategorized_primary = None
        for primary_attr in global_attributes:
            if primary_attr.primary_attribute.lower() == "uncategorized":
                uncategorized_primary = primary_attr
                break

        if not uncategorized_primary:
            uncategorized_primary = PrimaryAttribute(
                primary_attribute="Uncategorized", secondary_attributes=[]
            )
            global_attributes.append(uncategorized_primary)

        # Create a secondary attribute under 'Uncategorized' if needed
        uncategorized_secondary = None
        if uncategorized_primary.secondary_attributes:
            uncategorized_secondary = uncategorized_primary.secondary_attributes[0]
        else:
            uncategorized_secondary = SecondaryAttribute(
                attribute="Uncategorized", tertiary_attributes=[]
            )
            uncategorized_primary.secondary_attributes.append(uncategorized_secondary)

        # Create a tertiary attribute for unassigned statements
        uncategorized_tertiary = TertiaryAttribute(
            attribute="Uncategorized",
            statements=[
                Statement(statement=stmt, score=0.0) for stmt in unassigned_statements
            ],
        )
        uncategorized_secondary.tertiary_attributes.append(uncategorized_tertiary)

        logger.info(
            "Unassigned statements have been added to 'Uncategorized' attribute."
        )
    else:
        logger.info("All statements have been attributed.")

    # Check for duplicate statements assigned to multiple attributes
    statement_assignments = {}
    duplicates_found = False
    for primary_attr in global_attributes:
        for secondary_attr in primary_attr.secondary_attributes:
            for tertiary_attr in secondary_attr.tertiary_attributes:
                for stmt in tertiary_attr.statements:
                    stmt_text = stmt.statement.strip()
                    attr_path = (
                        primary_attr.primary_attribute,
                        secondary_attr.attribute,
                        tertiary_attr.attribute,
                    )
                    if stmt_text not in statement_assignments:
                        statement_assignments[stmt_text] = attr_path
                    else:
                        duplicates_found = True
                        logger.warning(
                            f"Duplicate statement detected: '{stmt_text}' is assigned to multiple attributes: "
                            f"{statement_assignments[stmt_text]} and {attr_path}"
                        )
                        # Optionally, remove the duplicate or decide which assignment to keep

    if not duplicates_found:
        logger.info("No duplicate statements found across attributes.")

    # Additional processing can include validating attribute names, ensuring no empty attributes, etc.
    # For now, we'll return the global_attributes as is.

    return global_attributes


def merge_attributes(attributes_list: List[PrimaryAttribute]) -> List[PrimaryAttribute]:
    """
    Merge attributes with the same names across a list of PrimaryAttribute objects.

    Args:
        attributes_list (List[PrimaryAttribute]): List of PrimaryAttribute objects.

    Returns:
        List[PrimaryAttribute]: Merged list of PrimaryAttribute objects.
    """
    primary_attr_map = {}

    for primary_attr in attributes_list:
        primary_name = primary_attr.primary_attribute.strip().lower()
        if primary_name not in primary_attr_map:
            primary_attr_map[primary_name] = primary_attr
        else:
            existing_primary = primary_attr_map[primary_name]
            # Merge secondary attributes
            secondary_attr_map = {
                sec.attribute.strip().lower(): sec
                for sec in existing_primary.secondary_attributes
            }
            for sec_attr in primary_attr.secondary_attributes:
                sec_name = sec_attr.attribute.strip().lower()
                if sec_name not in secondary_attr_map:
                    secondary_attr_map[sec_name] = sec_attr
                else:
                    existing_secondary = secondary_attr_map[sec_name]
                    # Merge tertiary attributes
                    tertiary_attr_map = {
                        ter.attribute.strip().lower(): ter
                        for ter in existing_secondary.tertiary_attributes
                    }
                    for ter_attr in sec_attr.tertiary_attributes:
                        ter_name = ter_attr.attribute.strip().lower()
                        if ter_name not in tertiary_attr_map:
                            tertiary_attr_map[ter_name] = ter_attr
                        else:
                            existing_tertiary = tertiary_attr_map[ter_name]
                            # Merge statements
                            existing_statements = {
                                stmt.statement for stmt in existing_tertiary.statements
                            }
                            for stmt in ter_attr.statements:
                                if stmt.statement not in existing_statements:
                                    existing_tertiary.statements.append(stmt)
                    existing_secondary.tertiary_attributes = list(
                        tertiary_attr_map.values()
                    )
            existing_primary.secondary_attributes = list(secondary_attr_map.values())

    return list(primary_attr_map.values())


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
                            "Score": stmt.score,
                        }
                    )

    # Serialize the data to JSON for inclusion in the prompt
    attribute_json = json.dumps(attribute_data, indent=2, ensure_ascii=False)
    weights_json = json.dumps(attribute_weights, indent=2, ensure_ascii=False)
    total_statements = len(attribute_data)

    # Initialize the Pydantic output parser for BusinessAnalysis
    business_analysis_parser = PydanticOutputParser(pydantic_object=BusinessAnalysis)
    business_analysis_format_instructions = (
        business_analysis_parser.get_format_instructions()
    )

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

Please provide a title for the report, and structure your analysis into five sections:
1. **Introduction**: Provide a background of the report, what it is based on, who it is for, and who commissioned it/led the project together with an overview of the business analysis using the details about the survey and the generally accepted standards for the {domain}.
2. **Introduction to Derived Customer Attributes**: Based on techniques used in QFD, provide an explanation on how customer attributes can be derived from a set of statements from the customers.
3. **Analysis**: Discuss key findings and insights derived from the data.
4. **Recommendations**: Offer actionable suggestions for the operations management based on the analysis with clear actionable to-do items.
5. **Conclusion**: A high-level conclusion of the analysis with a focus on customer satisfaction and quality.

**Instructions:**
- Ensure the analysis is professional, clear, and suitable for a business report aimed at the operational management team.
- Verify that all {total_statements} customer statements have been considered in your analysis.
- If you find any statements that are not addressed, please include them appropriately in the relevant sections.

Please return the results in the following format:
{business_analysis_format_instructions}
    """

    # Define retry parameters
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempt {attempt} to generate business analysis...")
            response = llm.invoke(prompt)
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
                    prompt = append_missing_statements_prompt(
                        prompt, missing_statements
                    )
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
                    title="Business Analysis",
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
                    title="Business Analysis",
                    introduction="Business analysis could not be generated due to an error.",
                    derived_attributes_introduction="",
                    analysis="",
                    recommendations="",
                    conclusions="",
                )

    # If all retries fail, return an incomplete BusinessAnalysis
    return BusinessAnalysis(
        title="Business Analysis",
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
