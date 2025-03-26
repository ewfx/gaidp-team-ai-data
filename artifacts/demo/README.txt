                                                       GEN AI-BASED DATA PROFILING & VALIDATION APP

Overview:

This Streamlit-based application extracts data validation rules from regulatory PDFs and applies them to CSV files. It supports both manual rule extraction and AI-powered extraction using the Hugging Face API.

Features:

- Extracts validation rules from PDFs (manual and AI-based).
- Generates Python validation code dynamically.
- Validates CSV data against extracted rules.
- Identifies invalid records and provides specific reasons.
- Offers detailed validation statistics.

Installation:

Prerequisites : Ensure you have Python installed (>=3.8) and the following dependencies:
- 'streamlit'
- 'pandas'
- 'pdfplumber'
- 'requests'

Setup:

1. Clone this repository or download the files.
2. Install required packages:
    - pip install streamlit pandas pdfplumber requests
3. Run the Streamlit app:
	- streamlit run GenAI_app.py

Usage:

1. Upload a Regulatory PDF 
   - The app extracts validation rules automatically.  
   - If an AI token (Hugging Face) is provided, AI-based extraction is attempted.  

2. Upload a CSV File
   - The app validates the data against the extracted rules.  
   - Displays valid and invalid records along with detailed validation statistics.  

3. Download Results  
   - Users can download validated and invalid records as CSV files.

Hugging Face Integration
- To enable AI-based rule extraction, get a token from [Hugging Face](https://huggingface.co/settings/tokens) and enter it in the app.
- The AI extracts rules in JSON format, which the app then converts into validation logic.

Error Handling
- If the app fails to extract rules, it falls back to AI-based extraction (if enabled).
- If validation fails due to an error in generated code, the error is displayed along with traceback information.

License
-This project is open-source and free to use.
