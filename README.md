# Customer Feedback Analysis

## Overview

This project analyzes customer feedback to derive key attributes, perform sentiment analysis, and generate comprehensive business reports.

## Setup Instructions

1. **Clone the Repository:**
   
   ```bash
   git clone https://github.com/yourusername/customer_feedback_analysis.git
   cd customer_feedback_analysis
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

        
        OPENAI_API_KEY=your_openai_api_key_here
        

5. **Prepare Data Files:**

    Ensure `survey.txt` and `survey_details.txt` are populated with relevant data.

### Usage
Run the main script with optional arguments:

```bash
python main.py --domain "Healthcare Services" --input_file "./survey.txt" --log_level "INFO"
--domain: The domain for analysis (default: "Healthcare Services").
--input_file: Path to the customer statements file (default: "./survey.txt").
--log_level: Set the logging level (choices: DEBUG, INFO, WARNING, ERROR, CRITICAL; default: INFO).
```

### Project Structure
* main.py: Entry point of the application.
* config.py: Configuration settings and logging.
* models.py: Pydantic models for data validation.
* data_loading.py: Functions to load data from files.
* chains.py: LangChain chain configurations.
* analysis.py: Data analysis functions.
* visualization.py: Visualization functions.
* report.py: Report generation functions.
* utils.py: Utility functions.
* requirements.txt: Python dependencies.
* .env: Environment variables.
* survey.txt: Customer statements.
* survey_details.txt: Survey details.
* README.md: Project overview and instructions.
