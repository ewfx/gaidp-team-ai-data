import streamlit as st
import pandas as pd
import pdfplumber
import json
import re
import traceback
import requests

# Initialize session state
if "profiling_rules" not in st.session_state:
    st.session_state.profiling_rules = {}
if "validation_code" not in st.session_state:
    st.session_state.validation_code = ""
if "hf_token" not in st.session_state:
    st.session_state.hf_token = ""
if "custom_prompt" not in st.session_state:
    st.session_state.custom_prompt = """Extract data validation rules from this document text. 
Focus on tables containing 'Field Name' and 'Allowable Values' columns.
Return a JSON object with field names as keys and validation rules as values.

Example output:
{
    "Field1": "numeric, 2 decimal places",
    "Field2": "YYYY-MM-DD date format",
    "Field3": "A, B, C or D"
}"""


# Function to escape strings safely
def escape_string(value):
    return repr(value)


# Function to safely escape column names
def escape_column_name(column_name):
    return f'"{column_name}"'


# AI Integration Functions
def verify_hf_token(hf_token: str) -> bool:
    """Basic format validation for Hugging Face tokens"""
    return hf_token.startswith("hf_") and len(hf_token) > 30

def query_hf_api(payload, hf_token):
    headers = {"Authorization": f"Bearer {hf_token}"}
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API request failed: {str(e)}")
        return None


def extract_rules_with_ai(text, hf_token, custom_prompt):
    prompt = f"""{custom_prompt}

Document text:
{text[:3000]}"""  # Limiting to first 3000 chars

    try:
        output = query_hf_api({
            "inputs": prompt,
            "parameters": {"max_new_tokens": 1000, "return_full_text": False}
        }, hf_token)

        if output and isinstance(output, list):
            result = output[0].get('generated_text', '{}')
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
        return {"Error": "AI extraction failed"}
    except Exception as e:
        return {"Error": f"AI error: {str(e)}"}


def extract_profiling_rules_from_pdf(pdf_file, custom_prompt=""):
    profiling_rules = {}
    reference_data = {}
    full_text = ""

    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                full_text += page_text + "\n\n"
                tables = page.extract_tables()

                for table in tables:
                    headers = [col.strip() if col else "" for col in table[0]]
                    if "Field Name" in headers and "Allowable Values" in headers:
                        fn_index = headers.index("Field Name")
                        av_index = headers.index("Allowable Values")

                        for row in table[1:]:
                            field_name = row[fn_index].strip() if row[fn_index] else ""
                            allowable_values = row[av_index].strip() if row[av_index] else ""

                            # Clean field names
                            field_name = re.sub(r'[^\w\s]', '', field_name)
                            field_name = re.sub(r'\s+', ' ', field_name).strip()
                            allowable_values = allowable_values.replace('\n', ' ').strip()

                            if field_name:
                                if "see " in allowable_values.lower():
                                    reference_data[field_name] = allowable_values
                                else:
                                    profiling_rules[field_name] = allowable_values

            # Resolve references
            for field, ref_text in reference_data.items():
                ref_section = ref_text.replace("see ", "").strip()
                if ref_section in profiling_rules:
                    profiling_rules[field] = profiling_rules[ref_section]

            # Only use AI if manual extraction found no rules and token is provided
            if not profiling_rules and st.session_state.hf_token:
                ai_rules = extract_rules_with_ai(full_text, st.session_state.hf_token, custom_prompt)
                if "Error" not in ai_rules:
                    profiling_rules.update(ai_rules)

        if not profiling_rules:
            return {"Error": "No clear validation rules found in the PDF."}
        return profiling_rules

    except Exception as e:
        return {"Error": f"Error extracting rules from PDF: {e}"}


def generate_validation_code(rules):
    code_lines = ["import pandas as pd", "import re", "\ndef validate_data(df):"]
    code_lines.append('    df["Validation Issues"] = ""')
    code_lines.append('    original_columns = df.columns.tolist()')

    for field, rule in rules.items():
        field_escaped = escape_column_name(field)
        rule_lower = rule.lower()

        # === Handle decimal format ===
        if "decimal format" in rule_lower:
            match = re.search(r'up to (\d+) decimal places', rule_lower)
            precision = int(match.group(1)) if match else 4
            regex = r"^-?\d+(\.\d{1,%d})?$" % precision
            code_lines.append(
                f'    mask = ~df[{field_escaped}].astype(str).str.match(r"{regex}", na=False)'
            )
            code_lines.append(
                f'    df.loc[mask, "Validation Issues"] += "Invalid decimal format in {field}; "'
            )

        # === Handle YYYY-MM-DD date format ===
        elif "yyyy-mm-dd" in rule_lower:
            code_lines.append(
                f'    converted = pd.to_datetime(df[{field_escaped}], format="%Y-%m-%d", errors="coerce")'
            )
            code_lines.append(
                f'    mask = converted.isnull()'
            )
            code_lines.append(
                f'    df.loc[mask, "Validation Issues"] += "Invalid date format in {field}; "'
            )

        # === Handle whole dollar amount ===
        elif ("whole dollar" in rule_lower or "no cents" in rule_lower or
              "no non-numeric formatting" in rule_lower or
              "no dollar sign" in rule_lower or "no commas" in rule_lower):
            regex = r"^-?\d+$"
            code_lines.append(
                f'    mask = ~df[{field_escaped}].astype(str).str.match(r"{regex}", na=False)'
            )
            code_lines.append(
                f'    df.loc[mask, "Validation Issues"] += "Invalid whole dollar format in {field}; "'
            )

        # === Handle negative sign enforcement ===
        elif "negative sign" in rule_lower and "parentheses" in rule_lower:
            code_lines.append(
                f'    mask = df[{field_escaped}].astype(str).str.contains(r"[()]", na=False)'
            )
            code_lines.append(
                f'    df.loc[mask, "Validation Issues"] += "Invalid negative format in {field} (parentheses not allowed); "'
            )

        # === Handle numeric ===
        elif rule_lower.strip() == "numeric":
            code_lines.append(
                f'    converted = pd.to_numeric(df[{field_escaped}], errors="coerce")'
            )
            code_lines.append(
                f'    mask = converted.isnull()'
            )
            code_lines.append(
                f'    df.loc[mask, "Validation Issues"] += "Non-numeric value in {field}; "'
            )

        else:
            values = [v.strip().rstrip('.') for v in re.split(r',\s*|;\s*|\. ', rule) if v.strip()]
            if len(values) > 1 and all(len(v) < 50 for v in values):
                values_escaped = [escape_string(v) for v in values]
                code_lines.append(
                    f'    mask = ~df[{field_escaped}].isin([{", ".join(values_escaped)}])'
                )
                code_lines.append(
                    f'    df.loc[mask, "Validation Issues"] += "Invalid value in {field}; "'
                )
            else:
                code_lines.append(f'    # Unable to interpret rule for {field}: {rule}')

    # Add statistics calculation
    code_lines.append("    # Calculate validation statistics")
    code_lines.append("    df_valid = df[df['Validation Issues'] == '']")
    code_lines.append("    df_invalid = df[df['Validation Issues'] != '']")
    code_lines.append("    total_records = len(df)")
    code_lines.append("    valid_count = len(df_valid)")
    code_lines.append("    invalid_count = len(df_invalid)")
    code_lines.append(
        "    anomaly_percentage = round((invalid_count / total_records * 100) if total_records > 0 else 0, 2)")
    code_lines.append("    stats = {")
    code_lines.append("        'total_records': total_records,")
    code_lines.append("        'valid_count': valid_count,")
    code_lines.append("        'invalid_count': invalid_count,")
    code_lines.append("        'anomaly_percentage': anomaly_percentage")
    code_lines.append("    }")
    code_lines.append("    return df_valid, df_invalid, stats\n")

    return "\n".join(code_lines)


# Streamlit App
st.title("🔍 Gen AI-Based Data Profiling & Validation")

# AI Configuration
st.session_state.hf_token = st.text_input(
    "Hugging Face API Token",
    type="password",
    help="Get your token from huggingface.co/settings/tokens"
)

if st.session_state.hf_token:
    if verify_hf_token(st.session_state.hf_token):
        st.success("Valid API token detected")
    else:
        st.warning("Token format appears invalid - will use manual extraction")

# Step 1: PDF Upload
st.subheader("Step 1: Upload Regulatory PDF to Extract Profiling Rules")
pdf_file = st.file_uploader("Upload a PDF file containing data validation rules", type=["pdf"])

# Show custom prompt only after file upload
if pdf_file is not None:
    st.session_state.custom_prompt = st.text_area(
        "Custom Prompt for AI",
        value=st.session_state.custom_prompt,
        height=200,
        help="Customize how the AI should extract rules from your document"
    )

if pdf_file is not None and st.button("Extract Profiling Rules"):
    with st.spinner("Extracting profiling rules..."):
        st.session_state.profiling_rules = extract_profiling_rules_from_pdf(
            pdf_file,
            st.session_state.custom_prompt
        )

    if "Error" in st.session_state.profiling_rules:
        st.error(st.session_state.profiling_rules["Error"])
    else:
        st.session_state.validation_code = generate_validation_code(st.session_state.profiling_rules)

        st.subheader("📜 Extracted Data Profiling Rules:")
        st.write(st.session_state.profiling_rules)

        st.download_button(
            label="Download Extracted Rules as JSON",
            data=json.dumps(st.session_state.profiling_rules, indent=2),
            file_name="profiling_rules.json",
            mime="application/json"
        )

        st.subheader("📝 Auto-Generated Python Validation Code:")
        st.code(st.session_state.validation_code)

# Step 2: Data Validation
st.subheader("Step 2: Upload Sample Data for Validation")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None and st.session_state.validation_code:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.replace(r'[^\w\s]', '', regex=True).str.replace(r'\s+', ' ', regex=True)

    st.write("🔹 Preview of Uploaded Data:")
    st.write(df.head())

    try:
        st.subheader("Generated Validation Code:")
        st.code(st.session_state.validation_code)

        exec_globals = {"df": df}
        exec(st.session_state.validation_code, exec_globals)
        df_validated, df_invalid, validation_stats = exec_globals["validate_data"](df)

        # Display validation statistics
        st.subheader("📊 Validation Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", validation_stats['total_records'])
        col2.metric("Valid Records", validation_stats['valid_count'],
                    delta=f"{validation_stats['valid_count'] / validation_stats['total_records'] * 100:.1f}% of total")
        col3.metric("Invalid Records", validation_stats['invalid_count'],
                    delta=f"{validation_stats['invalid_count'] / validation_stats['total_records'] * 100:.1f}% of total",
                    delta_color="inverse")

        st.metric("Anomaly Percentage",
                  f"{validation_stats['anomaly_percentage']}%",
                  delta=f"-{validation_stats['anomaly_percentage']}% from perfect" if validation_stats[
                                                                                          'anomaly_percentage'] > 0 else "0% anomalies",
                  delta_color="inverse")

        # Show Results
        st.subheader("🚨 Invalid Records with Reasons:")
        if not df_invalid.empty:
            st.write(df_invalid)
        else:
            st.write("No invalid records found.")

        st.subheader("✅ Valid Records:")
        st.write(df_validated)

        # Download buttons:
        st.download_button(
            label="Download Invalid Records with Reasons",
            data=df_invalid.to_csv(index=False),
            file_name="invalid_records_with_reasons.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Download Validated Data",
            data=df_validated.to_csv(index=False),
            file_name="validated_data.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error in validation code execution: {e}")
        st.text(traceback.format_exc())
        df_validated = df
        df_invalid = pd.DataFrame()
        validation_stats = {
            'total_records': len(df),
            'valid_count': len(df),
            'invalid_count': 0,
            'anomaly_percentage': 0.0
        }