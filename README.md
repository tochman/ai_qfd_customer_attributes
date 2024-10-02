
# Customer Feedback Analysis

## Overview

This project analyzes customer feedback to derive key attributes, perform sentiment analysis, classify the relevance of statements, and generate comprehensive business reports using Quality Function Deployment (QFD) methodologies.

## Setup Instructions

1. **Clone the Repository:**
   
   ```bash
   git clone https://github.com/tochman/ai_qfd_customer_attributes
   cd ai_qfd_customer_attributes
   ```

2. **Create a Virtual Environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set Up Environment Variables:**

    Create a `.env` file in the project root.

    Add your OpenAI API key:

    ```dotenv
    OPENAI_API_KEY=your_openai_api_key_here
    ```

5. **Prepare Data Files:**

    Ensure `survey.txt` and `survey_details.txt` are populated with relevant data.

    - **`survey.txt`**: A text file containing customer statements, one per line.
    - **`survey_details.txt`**: A text file containing details about the survey, such as objectives, methodology, and demographics.

## Usage

Run the main script with optional arguments:

```bash
python main.py --domain "Healthcare Services" --input_file "./survey.txt" --survey_details_file "./survey_details.txt" --log_level "INFO"
```

**Command-Line Arguments:**

- `--domain`: The domain for analysis (default: "Healthcare Services").
- `--input_file`: Path to the customer statements file (default: "./survey.txt").
- `--survey_details_file`: Path to the survey details file (default: "./survey_details.txt").
- `--log_level`: Set the logging level (choices: DEBUG, INFO, WARNING, ERROR, CRITICAL; default: INFO).

## Project Structure

- `main.py`: Entry point of the application.
- `config.py`: Configuration settings and logging setup.
- `models.py`: Pydantic models for data validation.
- `data_loading.py`: Functions to load data from files.
- `chains.py`: LangChain chain configurations, including relevance classification, sentiment analysis, and attribute derivation.
- `analysis.py`: Data analysis functions, such as calculating relative importance.
- `visualization.py`: Visualization functions for generating charts and graphs.
- `report.py`: Report generation functions.
- `utils.py`: Utility functions.
- `requirements.txt`: Python dependencies.
- `.env`: Environment variables (not committed to version control).
- `data/`: Directory containing data files.
  - `data/survey.txt`: Sample customer statements.
  - `data/survey_details.txt`: Sample survey details.

## Example

To analyze customer feedback in the domain of "Healthcare Services" using the provided `survey.txt` and `survey_details.txt` files:

```bash
python main.py --domain "Healthcare Services" --input_file "./data/survey.txt" --survey_details_file "./data/survey_details.txt" --log_level "INFO"
```

## Output

The script will generate a business report in docx format. 

## Logging

Logs are output to the console with the specified logging level. Adjust the logging level using the `--log_level` argument to get more or less verbose output.

## Notes

- Ensure your OpenAI API key has sufficient permissions and usage limits to handle the data volume.
- The script uses OpenAI's GPT-4 model; make sure your API key has access to this model.
- Be mindful of the token limits when processing large datasets. Adjust batch sizes in `config.py` if necessary.

## License

This project is licensed under the MIT License.
