# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import logging
import unicodedata
import re
import io
import traceback
import numpy as np

# PDF Parsing libraries
import pymupdf as fitz # PyMuPDF, Hapoalim & Credit Report
import pdfplumber # Leumi & Discount

from openai import OpenAI
from openai import APIError # Specific import for API errors
# Import specific OpenAI error types for more granular handling
# from openai import AuthenticationError, PermissionDeniedError, RateLimitError, APIConnectionError, InternalServerError

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- OpenAI Client Setup ---
client = None # Initialize client to None
try:
    # Attempt to get API key from secrets
    api_key = st.secrets["OPENAI_API_KEY"]
    if api_key: # Check if key exists and is not empty
       client = OpenAI(api_key=api_key)
       logging.info("OpenAI client initialized successfully.")
    else:
       logging.warning("OPENAI_API_KEY found in secrets but is empty.")
       st.error("מפתח OpenAI לא הוגדר כהלכה. שירות הצ'אט אינו זמין.")

except Exception as e:
    logging.error(f"Error loading OpenAI API key or initializing client: {e}", exc_info=True)
    st.error(f"שגיאה בטעינת מפתח OpenAI או בהפעלת שירות הצ'אט: {e}. הצ'אטבוט עשוי לא לפעול כראוי.")
    # Client remains None


# --- Helper Functions (Keep existing ones, assumed correct) ---
def clean_number_general(text):
    """Cleans numeric strings, handling currency symbols, commas, and parentheses."""
    if text is None: return None
    text = str(text).strip()
    text = re.sub(r'[₪,]', '', text)
    if text.startswith('(') and text.endswith(')'): text = '-' + text[1:-1]
    if text.endswith('-'): text = '-' + text[:-1]
    try:
        if text == "": return None # Handle empty string after cleaning
        return float(text)
    except ValueError:
        logging.debug(f"Could not convert '{text}' to float."); # Changed to debug to reduce log noise
        return None

def parse_date_general(date_str):
    """Parses date strings in multiple formats."""
    if date_str is None or pd.isna(date_str) or not isinstance(date_str, str): return None
    date_str = date_str.strip()
    if not date_str: return None
    try: return datetime.strptime(date_str, '%d/%m/%Y').date()
    except ValueError:
        try: return datetime.strptime(date_str, '%d/%m/%y').date()
        except ValueError:
            logging.debug(f"Could not parse date: {date_str}"); # Changed to debug
            return None

def normalize_text_general(text):
    """Normalizes Unicode text (removes potential hidden chars, ensures NFC)."""
    if text is None: return None
    text = str(text).replace('\r', ' ').replace('\n', ' ').replace('\u200b', '').strip()
    return unicodedata.normalize('NFC', text)

# --- PDF Parsers (HAPOALIM, LEUMI, DISCOUNT, CREDIT REPORT) ---
# Keep the parser functions as they were in the previous version.
# Added some debug logging within the parsers instead of info for lines that don't match patterns
# to reduce log noise unless debugging the parsers specifically.
# Ensured numeric columns are handled gracefully (fillna, errors='coerce') in parsers' output.

# --- HAPOALIM PARSER (Assume correct from previous version) ---
def extract_transactions_from_pdf_hapoalim(pdf_content_bytes, filename_for_logging="hapoalim_pdf"):
    """Extracts Date and Balance from Hapoalim PDF based on line patterns."""
    transactions = []
    try:
        doc = fitz.open(stream=pdf_content_bytes, filetype="pdf")
    except Exception as e:
        logging.error(f"Hapoalim: Failed to open/process PDF {filename_for_logging}: {e}", exc_info=True)
        return pd.DataFrame()

    date_pattern_end = re.compile(r"\s*(\d{1,2}/\d{1,2}/\d{4})\s*$")
    balance_pattern_start = re.compile(r"^\s*[₪]?[+\-]?\s*([\d,]+\.\d{2})")

    logging.info(f"Starting Hapoalim PDF parsing for {filename_for_logging}")

    for page_num, page in enumerate(doc):
        try:
            lines = page.get_text("text", sort=True).splitlines()
            for line_num, line_text in enumerate(lines):
                original_line = line_text
                line_normalized = normalize_text_general(original_line)

                if not line_normalized or len(line_normalized) < 10: continue

                date_match = date_pattern_end.search(original_line)
                if date_match:
                    date_str = date_match.group(1)
                    parsed_date = parse_date_general(date_str)

                    if parsed_date:
                        balance_match = balance_pattern_start.search(original_line)
                        if balance_match:
                            balance_str = balance_match.group(1)
                            balance = clean_number_general(balance_str)

                            balance_start_index = balance_match.start()
                            if balance_start_index > 0:
                                char_before = original_line[balance_start_index - 1]
                                if char_before not in (' ', '₪', '-', '+'):
                                    logging.debug(f"Hapoalim: Skipping line {line_num+1} due to unexpected char before balance: '{original_line[:balance_start_index].strip()}' -> '{original_line}'")
                                    continue

                            if balance is not None:
                                lower_line = line_normalized.lower()
                                if "יתרה לסוף יום" in lower_line or "עובר ושב" in lower_line or "תנועות בחשבון" in lower_line or "עמוד" in lower_line or "סך הכל" in lower_line or "הודעה זו כוללת" in lower_line:
                                    logging.debug(f"Hapoalim: Skipping potential header/footer/summary line: {original_line.strip()}")
                                    continue

                                transactions.append({
                                    'Date': parsed_date,
                                    'Balance': balance,
                                })
                                logging.debug(f"Hapoalim: Found transaction - Date: {parsed_date}, Balance: {balance}, Line: {original_line.strip()}")
                            # else: logging.debug(f"Hapoalim: Found date but failed to clean balance: {balance_str} in line: {original_line.strip()}")
                        # else: logging.debug(f"Hapoalim: Found date but no balance pattern match in line: {original_line.strip()}")
                    # else: logging.debug(f"Hapoalim: Found date pattern but failed to parse date string: {date_str} in line: {original_line.strip()}")
        except Exception as e:
            logging.error(f"Hapoalim: Error processing line {line_num+1} on page {page_num+1}: {e}", exc_info=True)
            continue

    doc.close()

    if not transactions:
        logging.warning(f"Hapoalim: No transactions found in {filename_for_logging}")
        return pd.DataFrame()

    df = pd.DataFrame(transactions)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce') # Ensure numeric, handle errors
    df = df.dropna(subset=['Date', 'Balance']) # Remove rows where date or balance parsing failed

    df = df.sort_values(by='Date').groupby('Date')['Balance'].last().reset_index()
    df = df.sort_values(by='Date').reset_index(drop=True) # Final sort

    logging.info(f"Hapoalim: Successfully extracted {len(df)} unique balance points from {filename_for_logging}")
    return df[['Date', 'Balance']]


# --- LEUMI PARSER (Assume correct from previous version) ---
def clean_transaction_amount_leumi(text):
    """Cleans Leumi transaction amount, handles potential unicode zero-width space."""
    if text is None or pd.isna(text) or text == '': return None
    text = str(text).strip().replace('₪', '').replace(',', '')
    text = text.lstrip('\u200b')
    if '.' not in text: return None # Requires a decimal point
    try:
        val = float(text)
        if abs(val) > 100_000_000:
             logging.debug(f"Leumi: Transaction amount seems excessively large: {val} from '{text}'. Skipping.")
             return None
        return val
    except ValueError:
        logging.debug(f"Leumi: Could not convert amount '{text}' to float.");
        return None

def clean_number_leumi(text):
     """Specific cleaner for Leumi numbers (balances often). Uses general cleaner."""
     return clean_number_general(text)


def parse_date_leumi(date_str):
    """Specific date parser for Leumi. Uses general parser."""
    return parse_date_general(date_str)

def normalize_text_leumi(text):
    """Normalizes Leumi text, including potential Hebrew reversal correction."""
    if text is None or pd.isna(text): return None
    text = str(text).replace('\r', ' ').replace('\n', ' ').replace('\u200b', '').strip()
    text = unicodedata.normalize('NFC', text)
    if any('\u0590' <= char <= '\u05EA' for char in text):
       words = text.split()
       reversed_text = ' '.join(words[::-1])
       return reversed_text
    return text

def parse_leumi_transaction_line_extracted_order_v2(line_text, previous_balance):
    """Attempts to parse a line assuming a specific column order from text extraction."""
    line = line_text.strip()
    if not line or len(line) < 15: return None
    pattern = re.compile(
        r"^([\-\u200b\d,\.]+)\s+" # 1: Balance
        r"(\d{1,3}(?:,\d{3})*\.\d{2})?\s*" # 2: Optional Amount
        r"(\S+)?\s*"             # 3: Optional single non-space char/code
        r"(.*?)\s+"              # 4: Description
        r"(\d{1,2}/\d{1,2}/\d{2,4})\s+" # 5: Value Date
        r"(\d{1,2}/\d{1,2}/\d{2,4})$"   # 6: Transaction Date
    )

    match = pattern.match(line)
    if not match: return None

    balance_str = match.group(1)
    amount_str = match.group(2)
    value_date_str = match.group(5)
    # transaction_date_str = match.group(6) # Not used for balance points

    parsed_date = parse_date_leumi(value_date_str)
    if not parsed_date: return None

    current_balance = clean_number_leumi(balance_str)
    if current_balance is None: return None

    amount = clean_transaction_amount_leumi(amount_str) # Can be None

    debit = None; credit = None
    if amount is not None and amount != 0 and previous_balance is not None:
        balance_diff = round(current_balance - previous_balance, 2)
        tolerance = 0.01
        if abs(balance_diff + amount) <= tolerance: debit = amount
        elif abs(balance_diff - amount) <= tolerance: credit = amount
        # else: logging.debug(f"Leumi: Balance change ({balance_diff}) does not match amount ({amount}) for line: {line}")

    return {'Date': parsed_date, 'Balance': current_balance, 'Debit': debit, 'Credit': credit}


def extract_leumi_transactions_line_by_line(pdf_content_bytes, filename_for_logging="leumi_pdf"):
    """Extracts Date and Balance from Leumi PDF by processing lines."""
    transactions_data = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_content_bytes)) as pdf:
            previous_balance = None
            found_first_balance = False
            logging.info(f"Starting Leumi PDF parsing for {filename_for_logging}")

            for page_num, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text(x_tolerance=2, y_tolerance=2, layout=True)
                    if not text: continue

                    lines = text.splitlines()
                    for line_num, line_text in enumerate(lines):
                        normalized_line = normalize_text_leumi(line_text.strip())
                        if not normalized_line or len(normalized_line) < 10: continue

                        if not found_first_balance:
                            initial_balance_match = re.search(r"(?:יתרה\s+קודמת|יתרת\s+סגירה\s+קודמת|יתרה\s+נכון\s+לתאריך)\s+([\-\u200b\d,\.]+)", normalized_line)
                            if initial_balance_match:
                                bal_str = initial_balance_match.group(1)
                                initial_bal = clean_number_leumi(bal_str)
                                if initial_bal is not None:
                                    previous_balance = initial_bal
                                    found_first_balance = True
                                    logging.debug(f"Leumi: Found initial balance on page {page_num+1}: {initial_bal} from line: {normalized_line.strip()}")
                                    continue

                            if previous_balance is None:
                                initial_entry = parse_leumi_transaction_line_extracted_order_v2(normalized_line, None)
                                if initial_entry and initial_entry['Balance'] is not None:
                                     previous_balance = initial_entry['Balance']
                                     found_first_balance = True
                                     logging.debug(f"Leumi: Treating first parsed entry balance as initial balance: {previous_balance} from line: {normalized_line.strip()}")
                                     # Don't continue here, let it potentially be added below

                        parsed_data = parse_leumi_transaction_line_extracted_order_v2(normalized_line, previous_balance)

                        if parsed_data:
                            current_balance = parsed_data['Balance']
                            parsed_date = parsed_data['Date']

                            if parsed_data['Debit'] is not None or parsed_data['Credit'] is not None or previous_balance is None:
                                if previous_balance is None: # This is the first valid balance line found
                                    previous_balance = current_balance

                                # Append only if different from the last entry or if it's the first entry
                                if not transactions_data or (transactions_data[-1]['Date'] != parsed_date or transactions_data[-1]['Balance'] != current_balance):
                                     transactions_data.append({'Date': parsed_date, 'Balance': current_balance})
                                     logging.debug(f"Leumi: Appended transaction balance - Date: {parsed_date}, Balance: {current_balance}, Line: {normalized_line.strip()}")

                                previous_balance = current_balance # Update previous balance for the next line
                            else:
                                # Line matched format but no transaction amount, update previous_balance if valid
                                if current_balance is not None:
                                     previous_balance = current_balance
                                logging.debug(f"Leumi: Matched line format but no transaction amount detected, only updating previous balance if valid: {normalized_line.strip()}")

                except Exception as e:
                     logging.error(f"Leumi: Error processing line {line_num+1} on page {page_num+1}: {e}", exc_info=True)
                     continue

    except Exception as e:
        logging.error(f"Leumi: FATAL ERROR processing PDF {filename_for_logging}: {e}", exc_info=True)
        return pd.DataFrame()

    if not transactions_data:
        logging.warning(f"Leumi: No transaction balances found in {filename_for_logging}")
        return pd.DataFrame()

    df = pd.DataFrame(transactions_data)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce') # Ensure numeric
    df = df.dropna(subset=['Date', 'Balance']) # Remove rows where date or balance parsing failed

    df = df.sort_values(by='Date').groupby('Date')['Balance'].last().reset_index()
    df = df.sort_values(by='Date').reset_index(drop=True) # Final sort

    logging.info(f"Leumi: Successfully extracted {len(df)} unique balance points from {filename_for_logging}")
    return df[['Date', 'Balance']]

# --- DISCOUNT PARSER (Assume correct from previous version) ---
def parse_discont_transaction_line(line_text):
    """Attempts to parse a line from Discount assuming specific date/balance placement."""
    line = line_text.strip()
    if not line or len(line) < 20: return None

    date_pattern = re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}/\d{1,2}/\d{2,4})$")
    date_match = date_pattern.search(line)
    if not date_match: return None

    line_before_dates = line[:date_match.start()].strip()
    if not line_before_dates: return None

    balance_pattern_start = re.compile(r"^[₪]?\s*([+\-]?[\d,]+\.\d{2})(?:\s+[₪]?\s*[+\-]?[\d,]+\.\d{2})?") # Primary pattern
    balance_match = balance_pattern_start.search(line_before_dates)

    if not balance_match:
         # Fallback for more flexible search
         balance_pattern_flexible = re.compile(r"^(?:.*?)\s*[₪]?\s*([+\-]?[\d,]+\.\d{2})(?:\s+[₪]?\s*[+\-]?[\d,]+\.\d{2})?")
         balance_match = balance_pattern_flexible.search(line_before_dates)
         if not balance_match:
            # logging.debug(f"Discount: Found dates but no clear balance pattern at start of '{line_before_dates}' from line: {line.strip()}")
            return None

    balance_str = balance_match.group(1)
    balance = clean_number_general(balance_str)

    if balance is None:
        logging.debug(f"Discount: Found dates but failed to clean balance: {balance_str} in line: {line.strip()}")
        return None

    date_str = date_match.group(1) # Use the first date found
    parsed_date = parse_date_general(date_str)

    if not parsed_date:
        logging.debug(f"Discount: Failed to parse date '{date_str}' from line: {line.strip()}")
        return None

    lower_line = line.lower()
    if any(phrase in lower_line for phrase in ["יתרת סגירה", "יתרה נכון", "סך הכל", "סהכ", "עמוד"]):
         logging.debug(f"Discount: Skipping likely closing balance/summary/footer line: {line.strip()}")
         return None
    if any(header_part in lower_line for header_part in ["תאריך רישום", "תאריך ערך", "תיאור", "אסמכתא", "סכום", "יתרה"]):
         logging.debug(f"Discount: Skipping likely header line: {line.strip()}")
         return None

    logging.debug(f"Discount: Parsed transaction - Date: {parsed_date}, Balance: {balance}, Line: {line.strip()}")
    return {'Date': parsed_date, 'Balance': balance}


def extract_and_parse_discont_pdf(pdf_content_bytes, filename_for_logging="discount_pdf"):
    """Extracts Date and Balance from Discount PDF by processing lines."""
    transactions = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_content_bytes)) as pdf:
            logging.info(f"Starting Discount PDF parsing for {filename_for_logging}")
            for page_num, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text(x_tolerance=2, y_tolerance=2, layout=True)
                    if text:
                        lines = text.splitlines()
                        for line_num, line_text in enumerate(lines):
                            normalized_line = normalize_text_general(line_text)
                            parsed = parse_discont_transaction_line(normalized_line)
                            if parsed:
                                # Avoid adding duplicate entries for the same date and balance
                                if not transactions or (transactions[-1]['Date'] != parsed['Date'] or transactions[-1]['Balance'] != parsed['Balance']):
                                     transactions.append(parsed)
                                # else: logging.debug(f"Discount: Skipping duplicate date/balance entry for line: {normalized_line.strip()}")
                            # else: logging.debug(f"Discount: Line did not match transaction pattern: {normalized_line.strip()}")

                except Exception as e:
                    logging.error(f"Discount: Error processing page {page_num+1}: {e}", exc_info=True)
                    continue

    except Exception as e:
        logging.error(f"Discount: FATAL ERROR processing PDF {filename_for_logging}: {e}", exc_info=True)
        return pd.DataFrame()

    if not transactions:
        logging.warning(f"Discount: No transaction balances found in {filename_for_logging}")
        return pd.DataFrame()

    df = pd.DataFrame(transactions)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce') # Ensure numeric
    df = df.dropna(subset=['Date', 'Balance']) # Remove rows with parsing errors

    df = df.sort_values(by='Date').groupby('Date')['Balance'].last().reset_index()
    df = df.sort_values(by='Date').reset_index(drop=True) # Final sort

    logging.info(f"Discount: Successfully extracted {len(df)} unique balance points from {filename_for_logging}")
    return df[['Date', 'Balance']]


# --- CREDIT REPORT PARSER (Assume correct from previous version) ---
COLUMN_HEADER_WORDS_CR = {
    "שם", "מקור", "מידע", "מדווח", "מזהה", "עסקה", "מספר", "עסקאות",
    "גובה", "מסגרת", "מסגרות", "סכום", "הלוואות", "מקורי", "יתרת", "חוב",
    "יתרה", "שלא", "שולמה", "במועד", "פרטי", "עסקה", "בנק", "אוצר", # Added "בנק", "אוצר"
    "סוג", "מטבע", "מניין", "ימים", "ריבית", "ממוצעת" # Add more potential headers
}
BANK_KEYWORDS_CR = {"בנק", "בע\"מ", "אגוד", "דיסקונט", "לאומי", "הפועלים", "מזרחי",
                 "טפחות", "הבינלאומי", "מרכנתיל", "אוצר", "החייל", "ירושלים",
                 "איגוד", "מימון", "ישיר", "כרטיסי", "אשראי", "מקס", "פיננסים",
                 "כאל", "ישראכרט", "פועלים", "לאומי", "דיסקונט", "מזרחי", "טפחות", "בינלאומי", "מרכנתיל", "איגוד"} # Added variations

def clean_credit_number(text):
    """Specific cleaner for credit report numbers, uses general."""
    return clean_number_general(text)

def process_entry_final_cr(entry_data, section, all_rows_list):
    """Processes a collected entry (bank name + numbers) into structured data."""
    if not entry_data or not entry_data.get('bank') or not entry_data.get('numbers'):
        logging.debug(f"CR: Skipping entry due to missing data: {entry_data}")
        return

    bank_name_raw = entry_data['bank']
    bank_name_cleaned = re.sub(r'\s*XX-[\w\d\-]+.*', '', bank_name_raw).strip()
    bank_name_cleaned = re.sub(r'\s+\d{1,3}(?:,\d{3})*$', '', bank_name_cleaned).strip()
    bank_name_cleaned = re.sub(r'\s+בע\"מ$', '', bank_name_cleaned, flags=re.IGNORECASE).strip()
    bank_name_cleaned = re.sub(r'\s+בנק$', '', bank_name_cleaned, flags=re.IGNORECASE).strip()
    bank_name_final = bank_name_cleaned if bank_name_cleaned else bank_name_raw

    is_likely_bank = any(kw in bank_name_final for kw in ["לאומי", "הפועלים", "דיסקונט", "מזרחי", "הבינלאומי", "מרכנתיל", "ירושלים", "איגוד", "טפחות", "אוצר"]) # Specific bank names
    if is_likely_bank and not bank_name_final.lower().endswith("בע\"מ"):
        bank_name_final += " בע\"מ"
    elif any(kw in bank_name_final for kw in ["מקס איט פיננסים", "מימון ישיר"]) and not bank_name_final.lower().endswith("בע\"מ"):
         bank_name_final += " בע\"מ"

    numbers_raw = entry_data['numbers']
    # Clean and filter out None values
    numbers = [clean_credit_number(n) for n in numbers_raw if clean_credit_number(n) is not None]

    num_count = len(numbers)
    limit_col, original_col, outstanding_col, unpaid_col = np.nan, np.nan, np.nan, np.nan

    if num_count >= 1: # Need at least one number
        val1 = numbers[0] if num_count > 0 else np.nan
        val2 = numbers[1] if num_count > 1 else np.nan
        val3 = numbers[2] if num_count > 2 else np.nan
        val4 = numbers[3] if num_count > 3 else np.nan

        if section in ["עו\"ש", "מסגרת אשראי"]:
             # Expected: Limit, Outstanding, Unpaid (optional)
             # Could be 2 numbers (Limit, Outstanding) or 3 (Limit, Outstanding, Unpaid)
             if num_count >= 2:
                  limit_col = val1
                  outstanding_col = val2
                  unpaid_col = val3 if num_count > 2 else 0.0
             elif num_count == 1: # If only one number, what is it? Assume Outstanding might be listed alone sometimes? Less likely.
                  # This might indicate malformed entry or a section with only one number field.
                  # Assuming the most common case is at least Limit+Outstanding (2 numbers).
                  # Log and skip or try best guess? Let's skip if less than 2 for these sections for safety.
                  logging.debug(f"CR: Skipping עו\"ש/מסגרת entry for '{bank_name_final}' with only 1 number.")
                  return


        elif section in ["הלוואה", "משכנתה"]:
            # Expected: Num Payments (optional), Original, Outstanding, Unpaid (optional)
            # Could be 2 numbers (Original, Outstanding), 3 (Original, Outstanding, Unpaid OR Num Payments, Original, Outstanding), or 4.
            if num_count >= 2:
                 # Check if the first number looks like a payment count (small integer)
                 if pd.notna(val1) and val1 == int(val1) and val1 > 0 and val1 < 600 and num_count >= 3: # Heuristic: payments > 0 and < 600 (50 years)
                      # Assuming Num Payments, Original, Outstanding, Unpaid
                      original_col = val2 if num_count > 1 else np.nan
                      outstanding_col = val3 if num_count > 2 else np.nan
                      unpaid_col = val4 if num_count > 3 else 0.0
                 else:
                     # Assuming Original, Outstanding, Unpaid (if 3+ numbers) or Original, Outstanding (if 2 numbers)
                     original_col = val1
                     outstanding_col = val2 if num_count > 1 else np.nan
                     unpaid_col = val3 if num_count > 2 else 0.0
            elif num_count == 1:
                 # If only one number, assume it's the outstanding balance? Less common, but possible.
                 # Let's assume it's the outstanding balance if that's all we get.
                 outstanding_col = val1
                 original_col = np.nan # Original is unknown
                 unpaid_col = 0.0 # Assume 0 unpaid if not listed
                 logging.debug(f"CR: Processing הלוואה/משכנתה entry for '{bank_name_final}' with only 1 number as Outstanding.")

        else: # Default case (e.g., "אחר" section) or fallback
            # Try to interpret the numbers based on quantity, assuming common loan-like structure
            if num_count >= 2:
                 original_col = val1
                 outstanding_col = val2
                 unpaid_col = val3 if num_count > 2 else 0.0
            elif num_count == 1:
                 outstanding_col = val1
                 original_col = np.nan
                 unpaid_col = 0.0
            # If num_count is 0, it was caught at the beginning of the function.
            # Limit is not applicable here (np.nan)
            logging.debug(f"CR: Processing 'אחר' entry for '{bank_name_final}' with {num_count} numbers.")


        # Append the processed row if at least outstanding or limit is not NaN
        if pd.notna(outstanding_col) or pd.notna(limit_col):
             all_rows_list.append({
                 "סוג עסקה": section,
                 "שם בנק/מקור": bank_name_final,
                 "גובה מסגרת": limit_col,
                 "סכום מקורי": original_col,
                 "יתרת חוב": outstanding_col,
                 "יתרה שלא שולמה": unpaid_col
             })
             logging.debug(f"CR: Appended row: {all_rows_list[-1]}")
        else:
            logging.debug(f"CR: Skipping entry for '{bank_name_final}' as no outstanding or limit found after number parsing.")


def extract_credit_data_final_v13(pdf_content_bytes, filename_for_logging="credit_report_pdf"):
    """Extracts structured credit data from the report PDF."""
    extracted_rows = []
    try:
        with fitz.open(stream=pdf_content_bytes, filetype="pdf") as doc:
            current_section = None
            current_entry = None
            last_line_was_id = False
            potential_bank_continuation_candidate = False

            section_patterns = {
                "חשבון עובר ושב": "עו\"ש",
                "הלוואה": "הלוואה",
                "משכנתה": "משכנתה",
                "מסגרת אשראי מתחדשת": "מסגרת אשראי",
                "אחר": "אחר" # Catch-all
            }
            number_line_pattern = re.compile(r"^\s*.*?(-?\d{1,3}(?:,\d{3})*\.?\d*)\s*.*?$") # Relaxed number pattern
            id_line_pattern = re.compile(r"^XX-[\w\d\-]+.*$")

            logging.info(f"Starting Credit Report PDF parsing for {filename_for_logging}")

            for page_num, page in enumerate(doc):
                try:
                    lines = page.get_text("text", sort=True).splitlines()
                    logging.debug(f"Page {page_num + 1} has {len(lines)} lines.")

                    for line_num, line_text in enumerate(lines):
                        line = normalize_text_general(line_text)
                        if not line: potential_bank_continuation_candidate = False; continue

                        is_section_header = False
                        for header_keyword, section_name in section_patterns.items():
                            if header_keyword in line and len(line) < len(header_keyword) + 25 and line.count(' ') < 6:
                                if current_entry and not current_entry.get('processed', False):
                                    process_entry_final_cr(current_entry, current_section, extracted_rows)
                                current_section = section_name
                                current_entry = None
                                last_line_was_id = False
                                potential_bank_continuation_candidate = False
                                is_section_header = True
                                logging.debug(f"CR: Detected section header: {line} -> {current_section}")
                                break
                        if is_section_header: continue

                        if line.startswith("סה\"כ") or line.startswith("הודעה זו כוללת") or "עמוד" in line:
                            if current_entry and not current_entry.get('processed', False):
                                process_entry_final_cr(current_entry, current_section, extracted_rows)
                            current_entry = None
                            last_line_was_id = False
                            potential_bank_continuation_candidate = False
                            logging.debug(f"CR: Detected summary/footer line: {line}")
                            continue

                        if current_section:
                            number_match = number_line_pattern.match(line)
                            if number_match:
                                if current_entry:
                                    try:
                                        number_str = number_match.group(1)
                                        number = clean_credit_number(number_str)
                                        if number is not None:
                                            num_list = current_entry.get('numbers', [])
                                            if last_line_was_id:
                                                if current_entry and not current_entry.get('processed', False):
                                                     process_entry_final_cr(current_entry, current_section, extracted_rows)
                                                current_entry = {'bank': current_entry['bank'], 'numbers': [number], 'processed': False}
                                                logging.debug(f"CR: Detected number after ID line, starting new entry for bank '{current_entry['bank']}' with first number: {number}")
                                            else:
                                                 if len(num_list) < 5:
                                                     current_entry['numbers'].append(number)
                                                     logging.debug(f"CR: Added number {number} to current entry for bank '{current_entry.get('bank', 'N/A')}'. Numbers: {current_entry['numbers']}")
                                                 else:
                                                     logging.debug(f"CR: Skipping extra number {number} for bank '{current_entry.get('bank', 'N/A')}'. Max numbers reached.")

                                    except Exception as e: # Catch potential errors during cleaning/appending
                                        logging.error(f"CR: Error processing number line '{line.strip()}': {e}", exc_info=True)

                                last_line_was_id = False
                                potential_bank_continuation_candidate = False
                                continue

                            is_id_line = id_line_pattern.match(line)
                            if is_id_line:
                                last_line_was_id = True
                                potential_bank_continuation_candidate = False
                                logging.debug(f"CR: Detected ID line: {line}")
                                continue

                            is_noise_line = any(word in line.split() for word in COLUMN_HEADER_WORDS_CR) or line in [':', '.', '-', '—'] or (len(line.replace(' ','')) < 3 and not line.replace(' ','').isdigit()) or re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", line) # Add date-only lines as noise
                            if is_noise_line:
                                last_line_was_id = False
                                potential_bank_continuation_candidate = False
                                logging.debug(f"CR: Skipping likely noise line: {line}")
                                continue

                            # If not a special line, treat as potential bank name or description
                            contains_number = any(char.isdigit() for char in line.replace(',', '').replace('.', '')) # Check for digits
                            contains_date_format = re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", line)
                            is_non_bank_phrase = any(phrase in line for phrase in ["סך הכל", "סהכ", "סה״כ", "ריכוז נתונים", "הודעה זו כוללת"]) # Add more phrases

                            if not contains_number and not contains_date_format and not is_non_bank_phrase:
                                 cleaned_line = re.sub(r'\s*XX-[\w\d\-]+.*|\s+\d+$', '', line).strip()
                                 common_continuations = ["לישראל", "בע\"מ", "ומשכנתאות", "נדל\"ן", "דיסקונט", "הראשון", "פיננסים", "איגוד", "אשראי", "חברה", "למימון", "שירותים"]
                                 seems_like_continuation_text = any(cleaned_line.startswith(cont) for cont in common_continuations) or (len(cleaned_line) > 3 and ' ' in cleaned_line)

                                 if potential_bank_continuation_candidate and current_entry and seems_like_continuation_text:
                                     current_entry['bank'] = (current_entry['bank'] + " " + cleaned_line).replace(" בע\"מ בע\"מ", " בע\"מ").strip()
                                     logging.debug(f"CR: Appended continuation '{cleaned_line}' to bank name. New bank name: '{current_entry['bank']}'")
                                     potential_bank_continuation_candidate = True # Still potentially continuing
                                 elif len(cleaned_line) > 3 and any(kw in cleaned_line for kw in BANK_KEYWORDS_CR):
                                      if current_entry and not current_entry.get('processed', False):
                                           process_entry_final_cr(current_entry, current_section, extracted_rows)
                                      current_entry = {'bank': cleaned_line, 'numbers': [], 'processed': False}
                                      potential_bank_continuation_candidate = True
                                      logging.debug(f"CR: Started new entry with bank name: '{cleaned_line}'")
                                 else: # Neither continuation nor new bank start
                                      if current_entry and current_entry.get('numbers') and not current_entry.get('processed', False):
                                          process_entry_final_cr(current_entry, current_section, extracted_rows)
                                          current_entry['processed'] = True
                                      potential_bank_continuation_candidate = False

                                 last_line_was_id = False
                            else: # Contains numbers/dates or is a known non-bank phrase
                                  if current_entry and current_entry.get('numbers') and not current_entry.get('processed', False):
                                       process_entry_final_cr(current_entry, current_section, extracted_rows)
                                       current_entry['processed'] = True
                                  last_line_was_id = False
                                  potential_bank_continuation_candidate = False

                except Exception as e:
                    logging.error(f"CR: Error processing line {line_num+1} on page {page_num+1}: {e}", exc_info=True)
                    continue

            if current_entry and not current_entry.get('processed', False):
                process_entry_final_cr(current_entry, current_section, extracted_rows)

    except Exception as e:
        logging.error(f"CreditReport: FATAL ERROR processing {filename_for_logging}: {e}", exc_info=True)
        return pd.DataFrame()

    if not extracted_rows:
        logging.warning(f"CreditReport: No structured entries found in {filename_for_logging}")
        return pd.DataFrame()

    df = pd.DataFrame(extracted_rows)

    final_cols = ["סוג עסקה", "שם בנק/מקור", "גובה מסגרת", "סכום מקורי", "יתרת חוב", "יתרה שלא שולמה"]
    for col in final_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[final_cols]

    for col in ["גובה מסגרת", "סכום מקורי", "יתרת חוב", "יתרה שלא שולמה"]:
        if col in df.columns:
             df[col] = pd.to_numeric(df[col], errors='coerce')
             if col == "יתרה שלא שולמה":
                  df[col] = df[col].fillna(0)

    df = df.dropna(subset=['גובה מסגרת', 'סכום מקורי', 'יתרת חוב', 'יתרה שלא שולמה'], how='all').reset_index(drop=True)

    logging.info(f"CreditReport: Successfully extracted {len(df)} entries from {filename_for_logging}")

    return df


# --- Initialize Session State ---
if 'app_stage' not in st.session_state: st.session_state.app_stage = "welcome"
if 'questionnaire_stage' not in st.session_state: st.session_state.questionnaire_stage = 0
if 'answers' not in st.session_state: st.session_state.answers = {}
if 'classification_details' not in st.session_state: st.session_state.classification_details = {}
if 'chat_messages' not in st.session_state: st.session_state.chat_messages = []
if 'df_bank_uploaded' not in st.session_state: st.session_state.df_bank_uploaded = pd.DataFrame()
if 'df_credit_uploaded' not in st.session_state: st.session_state.df_credit_uploaded = pd.DataFrame()
if 'bank_type_selected' not in st.session_state: st.session_state.bank_type_selected = "ללא דוח בנק"
if 'total_debt_from_credit_report' not in st.session_state: st.session_state.total_debt_from_credit_report = None
if 'uploaded_bank_file_name' not in st.session_state: st.session_state.uploaded_bank_file_name = None
if 'uploaded_credit_file_name' not in st.session_state: st.session_state.uploaded_credit_file_name = None


def reset_all_data():
    """Resets all session state variables to their initial state."""
    logging.info("Resetting all application data.")
    st.session_state.app_stage = "welcome"
    st.session_state.questionnaire_stage = 0
    st.session_state.answers = {}
    st.session_state.classification_details = {}
    st.session_state.chat_messages = []
    st.session_state.df_bank_uploaded = pd.DataFrame()
    st.session_state.df_credit_uploaded = pd.DataFrame()
    st.session_state.bank_type_selected = "ללא דוח בנק"
    st.session_state.total_debt_from_credit_report = None
    st.session_state.uploaded_bank_file_name = None
    st.session_state.uploaded_credit_file_name = None


# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="יועץ פיננסי משולב", page_icon="🧩")
st.title("🧩 יועץ פיננסי משולב: שאלון וניתוח דוחות")

# --- Sidebar ---
with st.sidebar:
    st.header("אפשרויות")
    if st.button("התחל מחדש את כל התהליך", key="reset_sidebar_button"):
        reset_all_data()
        st.rerun()
    st.markdown("---")
    st.caption("© כל הזכויות שמורות. כלי זה נועד למטרות אבחון ראשוני ואינו מהווה ייעוץ פיננסי.")


# --- Main Application Flow ---

if st.session_state.app_stage == "welcome":
    st.header("ברוכים הבאים ליועץ הפיננסי המשולב!")
    st.markdown("""
    כלי זה יעזור לך לקבל תמונה על מצבך הפיננסי באמצעות שילוב של:
    1.  **העלאת דוחות**: דוח בנק ודוח נתוני אשראי (אופציונלי, אך מומלץ לניתוח מדויק).
    2.  **שאלון פיננסי**: למילוי פרטים נוספים על הכנסות, הוצאות ומצבך הכללי.

    בסיום התהליך תקבל סיכום, סיווג פיננסי ראשוני, ויזואליזציות ואפשרות לשוחח עם יועץ וירטואלי.
    """)
    if st.button("התחל בהעלאת קבצים (מומלץ)", key="start_with_files"):
        st.session_state.app_stage = "file_upload"
        st.rerun()
    if st.button("התחל ישירות עם השאלון הפיננסי", key="start_with_questionnaire"):
        # Reset only questionnaire state if skipping files
        st.session_state.questionnaire_stage = 0
        st.session_state.answers = {}
        st.session_state.classification_details = {}
        st.session_state.total_debt_from_credit_report = None # Clear derived debt if skipping file step
        st.session_state.app_stage = "questionnaire"
        st.session_state.chat_messages = [] # Clear chat history
        st.rerun()


elif st.session_state.app_stage == "file_upload":
    st.header("שלב 1: העלאת דוחות")

    bank_type_options = ["ללא דוח בנק", "הפועלים", "דיסקונט", "לאומי"]
    current_bank_type_index = bank_type_options.index(st.session_state.bank_type_selected) if st.session_state.bank_type_selected in bank_type_options else 0
    st.session_state.bank_type_selected = st.selectbox(
        "בחר סוג דוח בנק:",
        bank_type_options,
        index=current_bank_type_index,
        key="bank_type_selector_main"
    )

    uploaded_bank_file = None
    if st.session_state.bank_type_selected != "ללא דוח בנק":
        uploaded_bank_file = st.file_uploader(f"העלה דוח בנק ({st.session_state.bank_type_selected}) (קובץ PDF)", type="pdf", key="bank_pdf_uploader_main")
        if uploaded_bank_file and st.session_state.get('uploaded_bank_file_name') != uploaded_bank_file.name:
             # Clear previously processed bank data if a new file is uploaded
             st.session_state.df_bank_uploaded = pd.DataFrame()
             st.session_state.uploaded_bank_file_name = uploaded_bank_file.name
             st.info(f"הקובץ {uploaded_bank_file.name} הועלה בהצלחה. לחץ על 'עבד קבצים' לעיבוד.")
        elif not uploaded_bank_file:
             st.session_state.uploaded_bank_file_name = None

    uploaded_credit_file = st.file_uploader("העלה דוח נתוני אשראי (קובץ PDF) (מומלץ)", type="pdf", key="credit_pdf_uploader_main")
    if uploaded_credit_file and st.session_state.get('uploaded_credit_file_name') != uploaded_credit_file.name:
         st.session_state.df_credit_uploaded = pd.DataFrame()
         st.session_state.total_debt_from_credit_report = None
         st.session_state.uploaded_credit_file_name = uploaded_credit_file.name
         st.info(f"הקובץ {uploaded_credit_file.name} הועלה בהצלחה. לחץ על 'עבד קבצים' לעיבוד.")
    elif not uploaded_credit_file:
         st.session_state.uploaded_credit_file_name = None


    if st.button("עבד קבצים והמשך לשאלון", key="process_files_button"):
        logging.info("Processing uploaded files...")
        processed_bank = False
        processed_credit = False
        error_processing = False

        with st.spinner("מעבד קבצים..."):
            # Process Bank File
            st.session_state.df_bank_uploaded = pd.DataFrame() # Reset before processing new file
            if uploaded_bank_file is not None and st.session_state.bank_type_selected != "ללא דוח בנק":
                try:
                    bank_file_bytes = uploaded_bank_file.getvalue()
                    parser_func = None
                    if st.session_state.bank_type_selected == "הפועלים": parser_func = extract_transactions_from_pdf_hapoalim
                    elif st.session_state.bank_type_selected == "לאומי": parser_func = extract_leumi_transactions_line_by_line
                    elif st.session_state.bank_type_selected == "דיסקונט": parser_func = extract_and_parse_discont_pdf

                    if parser_func:
                        st.session_state.df_bank_uploaded = parser_func(bank_file_bytes, uploaded_bank_file.name)

                    if st.session_state.df_bank_uploaded.empty:
                        st.warning(f"לא הצלחנו לחלץ נתונים מדוח הבנק ({st.session_state.bank_type_selected}). אנא וודא/י שהקובץ תקין והפורמט נתמך.")
                        error_processing = True
                    else:
                        st.success(f"דוח בנק ({st.session_state.bank_type_selected}) עובד בהצלחה!")
                        processed_bank = True
                except Exception as e:
                    logging.error(f"Error processing bank file {uploaded_bank_file.name}: {e}", exc_info=True)
                    st.error(f"אירעה שגיאה בעת עיבוד דוח הבנק: {e}")
                    error_processing = True


            # Process Credit File
            st.session_state.df_credit_uploaded = pd.DataFrame() # Reset before processing new file
            st.session_state.total_debt_from_credit_report = None # Reset
            if uploaded_credit_file is not None:
                try:
                    credit_file_bytes = uploaded_credit_file.getvalue()
                    st.session_state.df_credit_uploaded = extract_credit_data_final_v13(credit_file_bytes, uploaded_credit_file.name)
                    if st.session_state.df_credit_uploaded.empty:
                        st.warning("לא הצלחנו לחלץ נתונים מדוח האשראי. אנא וודא/י שהקובץ תקין.")
                        error_processing = True
                    else:
                        st.success("דוח נתוני אשראי עובד בהצלחה!")
                        processed_credit = True
                        if 'יתרת חוב' in st.session_state.df_credit_uploaded.columns:
                            total_debt = st.session_state.df_credit_uploaded['יתרת חוב'].fillna(0).sum()
                            st.session_state.total_debt_from_credit_report = total_debt
                            st.info(f"סך יתרת החוב שחושבה מדוח האשראי: {st.session_state.total_debt_from_credit_report:,.0f} ₪")
                        else:
                            st.warning("עמודת 'יתרת חוב' לא נמצאה בדוח האשראי המעובד.")

                except Exception as e:
                    logging.error(f"Error processing credit file {uploaded_credit_file.name}: {e}", exc_info=True)
                    st.error(f"אירעה שגיאה בעת עיבוד דוח נתוני האשראי: {e}")
                    error_processing = True

        # Move to questionnaire regardless of processing outcome
        if error_processing:
            st.warning("היו שגיאות בעיבוד חלק מהקבצים. הניתוח עשוי להיות חלקי.")

        st.session_state.app_stage = "questionnaire"
        st.session_state.questionnaire_stage = 0
        st.session_state.chat_messages = [] # Clear chat history when starting new questionnaire/analysis
        st.rerun()

    if st.button("דלג על העלאת קבצים והמשך לשאלון", key="skip_files_button"):
        logging.info("Skipping file upload and proceeding to questionnaire.")
        st.session_state.df_bank_uploaded = pd.DataFrame()
        st.session_state.df_credit_uploaded = pd.DataFrame()
        st.session_state.total_debt_from_credit_report = None
        st.session_state.bank_type_selected = "ללא דוח בנק"
        st.session_state.uploaded_bank_file_name = None
        st.session_state.uploaded_credit_file_name = None

        st.session_state.app_stage = "questionnaire"
        st.session_state.questionnaire_stage = 0
        st.session_state.chat_messages = []
        st.rerun()


elif st.session_state.app_stage == "questionnaire":
    st.header("שלב 2: שאלון פיננסי")
    st.markdown("אנא ענה/י על השאלות הבאות כדי לעזור לנו להבין טוב יותר את מצבך הפיננסי.")

    q_stage = st.session_state.questionnaire_stage

    # --- Questionnaire Stages ---

    # Stage 0: Initial Questions
    if q_stage == 0:
        st.subheader("חלק א': שאלות פתיחה")
        st.session_state.answers['q1_unusual_event'] = st.text_area("1. האם קרה משהו חריג שבגללו פנית?", value=st.session_state.answers.get('q1_unusual_event', ''), key="q_s0_q1")
        st.session_state.answers['q2_other_funding'] = st.text_area("2. האם יש מקורות מימון אחרים שבדקת?", value=st.session_state.answers.get('q2_other_funding', ''), key="q_s0_q2")

        existing_loans_bool_key = 'q3_existing_loans_bool_radio'
        default_loan_bool_index = ("לא","כן").index(st.session_state.answers.get(existing_loans_bool_key, 'לא'))
        st.session_state.answers[existing_loans_bool_key] = st.radio(
            "3. האם קיימות הלוואות נוספות (לא משכנתא)?",
            ("כן", "לא"),
            index=default_loan_bool_index,
            key="q_s0_q3_bool"
        )
        if st.session_state.answers[existing_loans_bool_key] == "כן":
            st.session_state.answers['q3_loan_repayment_amount'] = st.number_input(
                "מה גובה ההחזר החודשי הכולל עליהן?",
                min_value=0.0, value=float(st.session_state.answers.get('q3_loan_repayment_amount', 0.0)), step=100.0, key="q_s0_q3_amount"
            )
        else: st.session_state.answers['q3_loan_repayment_amount'] = 0.0

        balanced_bool_key = 'q4_financially_balanced_bool_radio'
        default_balanced_index = ("כן","בערך","לא").index(st.session_state.answers.get(balanced_bool_key, 'כן'))
        st.session_state.answers[balanced_bool_key] = st.radio(
            "4. האם אתם מאוזנים כלכלית כרגע (הכנסות מכסות הוצאות)?",
            ("כן", "בערך", "לא"),
            index=default_balanced_index,
            key="q_s0_q4_bool"
        )
        st.session_state.answers['q4_situation_change_next_year'] = st.text_area("האם המצב הכלכלי צפוי להשתנות משמעותית בשנה הקרובה (לחיוב או לשלילה)?", value=st.session_state.answers.get('q4_situation_change_next_year', ''), key="q_s0_q4_change")

        if st.button("הבא", key="q_s0_next"):
            st.session_state.questionnaire_stage += 1
            st.rerun()

    # Stage 1: Income
    elif q_stage == 1:
        st.subheader("חלק ב': הכנסות (נטו חודשי)")
        st.session_state.answers['income_employee'] = st.number_input("הכנסתך (נטו):", min_value=0.0, value=float(st.session_state.answers.get('income_employee', 0.0)), step=100.0, key="q_s1_inc_emp")
        st.session_state.answers['income_partner'] = st.number_input("הכנסת בן/בת הזוג (נטו):", min_value=0.0, value=float(st.session_state.answers.get('income_partner', 0.0)), step=100.0, key="q_s1_inc_partner")
        st.session_state.answers['income_other'] = st.number_input("הכנסות נוספות (קצבאות, שכר דירה וכו'):", min_value=0.0, value=float(st.session_state.answers.get('income_other', 0.0)), step=100.0, key="q_s1_inc_other")

        total_net_income = sum(float(st.session_state.answers.get(k,0.0)) for k in ['income_employee','income_partner','income_other'])
        st.session_state.answers['total_net_income'] = total_net_income
        st.metric("סך הכנסות נטו (חודשי):", f"{total_net_income:,.0f} ₪")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("הקודם", key="q_s1_prev"): st.session_state.questionnaire_stage -= 1; st.rerun()
        with col2:
            if st.button("הבא", key="q_s1_next"): st.session_state.questionnaire_stage += 1; st.rerun()

    # Stage 2: Fixed Expenses
    elif q_stage == 2:
        st.subheader("חלק ג': הוצאות קבועות חודשיות")
        st.session_state.answers['expense_rent_mortgage'] = st.number_input("שכירות / החזר משכנתא:", min_value=0.0, value=float(st.session_state.answers.get('expense_rent_mortgage', 0.0)), step=100.0, key="q_s2_exp_rent")
        default_debt_repayment = float(st.session_state.answers.get('q3_loan_repayment_amount', 0.0))
        st.session_state.answers['expense_debt_repayments'] = st.number_input(
            "החזרי הלוואות נוספות (לא משכנתא, כולל כרטיסי אשראי אם יש החזר קבוע):",
            min_value=0.0, value=float(st.session_state.answers.get('expense_debt_repayments', default_debt_repayment)), step=100.0, key="q_s2_exp_debt"
        )
        st.session_state.answers['expense_alimony_other'] = st.number_input("מזונות / הוצאות קבועות גדולות אחרות (למשל: חסכון קבוע, ביטוחים גבוהים):", min_value=0.0, value=float(st.session_state.answers.get('expense_alimony_other', 0.0)), step=100.0, key="q_s2_exp_alimony")

        total_fixed_expenses = sum(float(st.session_state.answers.get(k,0.0)) for k in ['expense_rent_mortgage','expense_debt_repayments','expense_alimony_other'])
        st.session_state.answers['total_fixed_expenses'] = total_fixed_expenses
        st.metric("סך הוצאות קבועות:", f"{total_fixed_expenses:,.0f} ₪")

        total_net_income = float(st.session_state.answers.get('total_net_income', 0.0))
        monthly_balance = total_net_income - total_fixed_expenses
        st.session_state.answers['monthly_balance'] = monthly_balance
        st.metric("יתרה פנויה חודשית (הכנסות פחות קבועות):", f"{monthly_balance:,.0f} ₪")
        if monthly_balance < 0: st.warning("שימו לב: ההוצאות הקבועות גבוהות מההכנסות נטו.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("הקודם", key="q_s2_prev"): st.session_state.questionnaire_stage -= 1; st.rerun()
        with col2:
            if st.button("הבא", key="q_s2_next"): st.session_state.questionnaire_stage += 1; st.rerun()

    # Stage 3: Total Debts & Arrears
    elif q_stage == 3:
        st.subheader("חלק ד': חובות ופיגורים")

        default_total_debt = float(st.session_state.answers.get('total_debt_amount', 0.0))
        if st.session_state.total_debt_from_credit_report is not None:
            default_total_debt = st.session_state.total_debt_from_credit_report
            st.info(f"סך יתרת החוב שחושבה מדוח האשראי שהועלה הוא: {st.session_state.total_debt_from_credit_report:,.0f} ₪. **ניתן לעדכן את הסכום למטה אם קיימים חובות נוספים שלא מופיעים בדוח.**")
        else:
             st.info("אנא הזן/י את סך כל החובות הקיימים (למעט משכנתא).")


        st.session_state.answers['total_debt_amount'] = st.number_input(
            "מה היקף החובות הכולל שלך (למעט משכנתא)?",
            min_value=0.0, value=float(st.session_state.answers.get('total_debt_amount', default_total_debt)), step=100.0, key="q_s3_total_debt"
        )

        arrears_key = 'arrears_collection_proceedings_radio'
        default_arrears_index = ("לא","כן").index(st.session_state.answers.get(arrears_key, 'לא'))
        st.session_state.answers[arrears_key] = st.radio(
            "האם קיימים פיגורים משמעותיים בתשלומים או הליכי גבייה פעילים נגדך?",
            ("כן", "לא"),
            index=default_arrears_index,
            key="q_s3_arrears"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("הקודם", key="q_s3_prev"): st.session_state.questionnaire_stage -= 1; st.rerun()
        with col2:
            if st.button("סיום שאלון וקבלת סיכום", key="q_s3_next_finish"):
                current_total_debt = float(st.session_state.answers.get('total_debt_amount', 0.0))
                current_total_net_income = float(st.session_state.answers.get('total_net_income', 0.0))

                annual_income = current_total_net_income * 12
                st.session_state.answers['annual_income'] = annual_income

                if annual_income > 0:
                     st.session_state.answers['debt_to_income_ratio'] = current_total_debt / annual_income
                else:
                     st.session_state.answers['debt_to_income_ratio'] = float('inf') if current_total_debt > 0 else 0.0

                ratio = st.session_state.answers['debt_to_income_ratio']
                arrears_exist = st.session_state.answers.get(arrears_key, 'לא') == 'כן'

                classification = "לא נקבע"
                description = "לא הושלם סיווג ראשוני."
                color = "gray"
                next_stage = "summary"

                if arrears_exist:
                    classification = "אדום"
                    description = "קיימים פיגורים משמעותיים או הליכי גבייה פעילים."
                    color = "red"
                    next_stage = "summary"

                elif ratio < 1:
                    classification = "ירוק"
                    description = "סך החוב נמוך משמעותית מההכנסה השנתית (פחות משנת הכנסה)."
                    color = "green"
                    next_stage = "summary"

                elif 1 <= ratio <= 2:
                    classification = "צהוב (בבדיקה)"
                    description = "סך החוב בגובה ההכנסה של 1-2 שנים."
                    color = "orange"
                    next_stage = 100 # Go to special intermediate stage for Yellow

                else: # ratio > 2
                    classification = "אדום"
                    description = "סך החוב גבוה משמעותית מההכנסה השנתית (מעל שנתיים הכנסה)."
                    color = "red"
                    next_stage = "summary"

                st.session_state.classification_details = {
                    'classification': classification,
                    'description': description,
                    'color': color
                }

                if next_stage == "summary":
                    st.session_state.app_stage = "summary"
                    st.session_state.questionnaire_stage = -1 # Indicate questionnaire is finished
                else:
                    st.session_state.questionnaire_stage = next_stage

                st.rerun()

    # Stage 100: Intermediate questions for Yellow classification
    elif q_stage == 100:
        st.subheader("שאלות הבהרה נוספות")
        st.warning(f"תוצאות ראשוניות: יחס החוב להכנסה שלך הוא {st.session_state.answers.get('debt_to_income_ratio', 0.0):.2f}. ({st.session_state.classification_details.get('description')})")

        arrears_exist = st.session_state.answers.get('arrears_collection_proceedings_radio', 'לא') == 'כן'

        if arrears_exist:
             st.error("נמצא שקיימים הליכי גבייה. מצב זה מסווג אוטומטית כ'אדום'.")
             st.session_state.classification_details.update({'classification': "אדום", 'description': st.session_state.classification_details.get('description','') + " קיימים הליכי גבייה.", 'color': "red"})
             if st.button("המשך לסיכום", key="q_s100_to_summary_red_recheck"):
                 st.session_state.app_stage = "summary"
                 st.session_state.questionnaire_stage = -1
                 st.rerun()
        else:
            total_debt = float(st.session_state.answers.get('total_debt_amount', 0.0))
            fifty_percent_debt = total_debt * 0.5 if total_debt > 0 else 0.0
            st.session_state.answers['can_raise_50_percent_radio'] = st.radio(
                f"האם תוכל/י לגייס סכום השווה לכ-50% מסך החובות הלא מגובים במשכנתא ({fifty_percent_debt:,.0f} ₪) ממקורות תמיכה (משפחה, חברים, מימוש נכסים) תוך זמן סביר (עד מספר חודשים)?",
                ("כן", "לא"),
                index=("לא","כן").index(st.session_state.answers.get('can_raise_50_percent_radio', 'לא')),
                key="q_s100_q_raise_funds"
            )
            if st.button("המשך לסיכום", key="q_s100_to_summary_yellow_check"):
                if st.session_state.answers.get('can_raise_50_percent_radio', 'לא') == "כן":
                    st.session_state.classification_details.update({'classification': "צהוב", 'description': st.session_state.classification_details.get('description','') + " אין הליכי גבייה ויש יכולת לגייס 50% מהחוב ממקורות תמיכה.", 'color': "orange"})
                else:
                    st.session_state.classification_details.update({'classification': "צהוב+", 'description': st.session_state.classification_details.get('description','') + " אין הליכי גבייה אך אין יכולת מיידית לגייס סכום משמעותי ממקורות תמיכה.", 'color': "orange"}) # Maybe make this "צהוב+" or keep red? Let's use orange for now but add nuance. Or maybe it *is* red without ability to raise funds? Revert to Red if no ability to raise funds seems more appropriate for actionability.

                # Re-evaluating classification for yellow based on ability to raise funds (simplified)
                if st.session_state.answers.get('can_raise_50_percent_radio', 'לא') == "כן":
                     st.session_state.classification_details.update({'classification': "צהוב", 'description': "סך החוב בגובה ההכנסה של 1-2 שנים, אין הליכי גבייה ויש יכולת לגייס 50% מהחוב ממקורות תמיכה.", 'color': "orange"})
                else:
                     st.session_state.classification_details.update({'classification': "אדום", 'description': "סך החוב בגובה ההכנסה של 1-2 שנים, אין הליכי גבייה אך **אין** יכולת לגייס 50% מהחוב ממקורות תמיכה.", 'color': "red"}) # Leaning towards red if significant external help isn't possible for a yellow case

                st.session_state.app_stage = "summary"
                st.session_state.questionnaire_stage = -1
                st.rerun()

        if st.button("חזור לשלב הקודם בשאלון", key="q_s100_prev"):
            st.session_state.questionnaire_stage = 3; st.rerun()


elif st.session_state.app_stage == "summary":
    st.header("שלב 3: סיכום, ויזואליזציות וייעוץ")
    st.markdown("להלן סיכום הנתונים שאספנו והניתוח הראשוני.")

    # Retrieve calculated metrics
    total_net_income_ans = float(st.session_state.answers.get('total_net_income', 0.0))
    total_fixed_expenses_ans = float(st.session_state.answers.get('expense_rent_mortgage', 0.0)) + float(st.session_state.answers.get('expense_debt_repayments', 0.0)) + float(st.session_state.answers.get('expense_alimony_other', 0.0))
    monthly_balance_ans = total_net_income_ans - total_fixed_expenses_ans
    total_debt_amount_ans = float(st.session_state.answers.get('total_debt_amount', 0.0))
    annual_income_ans = total_net_income_ans * 12
    debt_to_income_ratio_ans = (total_debt_amount_ans / annual_income_ans) if annual_income_ans > 0 else (float('inf') if total_debt_amount_ans > 0 else 0.0)


    st.subheader("📊 סיכום נתונים פיננסיים")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("💰 סך הכנסות נטו (חודשי)", f"{total_net_income_ans:,.0f} ₪")
        st.metric("💸 סך הוצאות קבועות (חודשי)", f"{total_fixed_expenses_ans:,.0f} ₪")

    with col2:
        st.metric("📊 יתרה פנויה (חודשי)", f"{monthly_balance_ans:,.0f} ₪")
        st.metric("📈 הכנסה שנתית", f"{annual_income_ans:,.0f} ₪")

    with col3:
        st.metric("🏦 סך חובות (ללא משכנתא)", f"{total_debt_amount_ans:,.0f} ₪")
        # Check if credit report debt exists and is different from questionnaire debt
        if st.session_state.total_debt_from_credit_report is not None and abs(st.session_state.total_debt_from_credit_report - total_debt_amount_ans) > 1:
             st.caption(f"(מדוח אשראי שנותח: {st.session_state.total_debt_from_credit_report:,.0f} ₪)")
        st.metric("⚖️ יחס חוב להכנסה שנתית", f"{debt_to_income_ratio_ans:.2%}")


    # Display classification and recommendations
    st.subheader("סיווג מצב פיננסי והמלצה ראשונית:")
    classification = st.session_state.classification_details.get('classification', "לא נקבע")
    description = st.session_state.classification_details.get('description', "")
    color = st.session_state.classification_details.get('color', "gray")

    if color == "green":
        st.success(f"🟢 **סיווג: {classification}**")
        st.markdown("""
        **מצב יציב.** יחס החוב להכנסה נמוך. זהו מצב המאפשר גמישות פיננסית.
        * **המלצה ראשונית:** המשך/י בניהול פיננסי אחראי. כדאי לשקול הגדלת חיסכון או השקעות. דוח האשראי יכול לעזור להבין את המגבלות הקיימות ולשפר תנאים עתידיים.
        """)
    elif color == "orange":
        st.warning(f"🟡 **סיווג: {classification}**")
        st.markdown("""
        **מצב הדורש בדיקה ותשומת לב.** יחס החוב להכנסה מעיד על פוטנציאל קושי, אך אין הליכי גבייה ויש יכולת לגייס סכום משמעותי בחירום.
        * **המלצה ראשונית:** מומלץ לבחון לעומק את פירוט החובות (בדוח האשראי) וההוצאות (דרך דוח הבנק או מעקב אישי). בנה/י תוכנית פעולה ממוקדת לצמצום החובות. הגדלת הכנסות או קיצוץ בהוצאות לא חיוניות יכולים לעזור משמעותית. השתמש/י בצ'אט כדי לבקש רעיונות לניהול תקציב או סדר עדיפויות בחובות.
        """)
    elif color == "red":
        st.error(f"🔴 **סיווג: {classification}**")
        st.markdown("""
        **מצב קשה הדורש התערבות מיידית.** יחס החוב להכנסה גבוה או שקיימים הליכי גבייה או שאין יכולת לגייס סכום משמעותי בחירום. המצב דורש טיפול דחוף.
        * **המלצה ראשונית:** אל תדחה/י זאת! פנה/י בהקדם לייעוץ מקצועי בתחום כלכלת המשפחה והחובות. ארגונים כמו "פעמונים" או יועצים פרטיים מומחים יכולים לעזור בבניית תוכנית חירום, ניהול משא ומתן עם נושים, ובחינת אפשרויות משפטיות אם נדרש. חשוב להבין את מלוא היקף החוב ולהפסיק לצבור חוב חדש.
        """)
    else:
         st.info(f"⚫ **סיווג: {classification}**")
         st.markdown("""
         **הסיווג לא הושלם.** ייתכן שחסרים נתונים בשאלון.
         * **המלצה ראשונית:** אנא השלם/י את השאלון כדי לקבל סיווג והמלצה ראשונית.
         """)

    st.markdown("---")
    st.subheader("🎨 ויזואליזציות מרכזיות")

    # Visualization 1: Debt Breakdown from Credit Report (Pie Chart)
    if not st.session_state.df_credit_uploaded.empty and 'סוג עסקה' in st.session_state.df_credit_uploaded.columns and 'יתרת חוב' in st.session_state.df_credit_uploaded.columns:
        df_credit_cleaned = st.session_state.df_credit_uploaded.copy()
        df_credit_cleaned['יתרת חוב_numeric'] = pd.to_numeric(df_credit_cleaned['יתרת חוב'], errors='coerce').fillna(0)
        debt_summary = df_credit_cleaned.groupby("סוג עסקה")["יתרת חוב_numeric"].sum().reset_index()
        debt_summary = debt_summary[debt_summary['יתרת חוב_numeric'] > 0]

        if not debt_summary.empty:
            fig_debt_pie = px.pie(
                debt_summary,
                values='יתרת חוב_numeric',
                names='סוג עסקה',
                title='פירוט יתרות חוב (מדוח נתוני אשראי)',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_debt_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_debt_pie, use_container_width=True)
        else:
             st.info("אין נתוני חוב משמעותיים בדוח האשראי להצגה.")

    # Visualization 2: Debt vs. Income (Bar Chart)
    if total_debt_amount_ans > 0 or annual_income_ans > 0 :
        comparison_data = pd.DataFrame({
            'קטגוריה': ['סך חובות (ללא משכנתא)', 'הכנסה שנתית'],
            'סכום': [total_debt_amount_ans, annual_income_ans]
        })
        fig_debt_income_bar = px.bar(
            comparison_data,
            x='קטגוריה',
            y='סכום',
            title='השוואת סך חובות להכנסה שנתית',
            color='קטגוריה',
            text_auto=True,
            labels={'קטגוריה': '', 'סכום': 'סכום ב₪'}
        )
        fig_debt_income_bar.update_layout(yaxis_tickformat='~s')
        st.plotly_chart(fig_debt_income_bar, use_container_width=True)
    else:
         st.info("אין נתוני חוב או הכנסה להצגת השוואה.")


    # Visualization 3: Bank Balance Trend (Line Chart)
    if not st.session_state.df_bank_uploaded.empty:
        st.subheader(f"מגמת יתרת חשבון בנק ({st.session_state.bank_type_selected})")
        df_bank_plot = st.session_state.df_bank_uploaded.dropna(subset=['Date', 'Balance']).sort_values(by='Date').reset_index(drop=True)
        if not df_bank_plot.empty:
            fig_balance_trend = px.line(
                df_bank_plot,
                x='Date',
                y='Balance',
                title=f'מגמת יתרת חשבון בנק',
                markers=True
            )
            fig_balance_trend.update_layout(yaxis_tickformat='~s')
            st.plotly_chart(fig_balance_trend, use_container_width=True)
        else:
             st.info(f"אין נתוני יתרות תקינים בדוח הבנק ({st.session_state.bank_type_selected}) להצגה.")
    elif st.session_state.bank_type_selected != "ללא דוח בנק":
        st.info(f"לא הועלה או לא הצלחנו לעבד דוח בנק מסוג {st.session_state.bank_type_selected}.")
    else:
         st.info("לא נבחר סוג דוח בנק או לא הועלה קובץ.")


    # Display DataFrames (optional expander)
    with st.expander("הצג נתונים גולמיים שחולצו מדוחות שהועלו"):
        if not st.session_state.df_credit_uploaded.empty:
            st.write("נתוני אשראי מחולצים:")
            styled_credit_df = st.session_state.df_credit_uploaded.style.format({
                'גובה מסגרת': "{:,.0f}", 'סכום מקורי': "{:,.0f}",
                'יתרת חוב': "{:,.0f}", 'יתרה שלא שולמה': "{:,.0f}"
            })
            st.dataframe(styled_credit_df, use_container_width=True)
        else: st.write("לא הועלה או לא עובד דוח נתוני אשראי.")

        st.markdown("---")

        if not st.session_state.df_bank_uploaded.empty:
            st.write(f"נתוני יתרות בנק מחולצים ({st.session_state.bank_type_selected}):")
            styled_bank_df = st.session_state.df_bank_uploaded.style.format({"Balance": '{:,.2f}'})
            st.dataframe(styled_bank_df, use_container_width=True)
        else:
             if st.session_state.bank_type_selected != "ללא דוח בנק": st.write(f"לא הועלה או לא עובד דוח בנק מסוג {st.session_state.bank_type_selected}.")
             else: st.write("לא נבחר או הועלה דוח בנק.")


    st.markdown("---")
    # --- Chatbot Interface ---
    st.header("💬 צ'אט עם יועץ פיננסי וירטואלי")
    if client:
        st.markdown("שאל/י כל שאלה על מצבך הפיננסי, הנתונים שהוצגו, או כלכלת המשפחה.")

        # Prepare context for chatbot
        financial_context = "סיכום המצב הפיננסי של המשתמש:\n"
        financial_context += f"- סך הכנסות נטו חודשיות (משאלון): {total_net_income_ans:,.0f} ₪\n"
        financial_context += f"- סך הוצאות קבועות חודשיות (משאלון): {total_fixed_expenses_ans:,.0f} ₪\n"
        financial_context += f"- מאזן חודשי (יתרה פנויה): {monthly_balance_ans:,.0f} ₪\n"
        financial_context += f"- סך חובות (ללא משכנתא, לאחר שאלון ואולי עדכון מדוח): {total_debt_amount_ans:,.0f} ₪\n"

        # Add credit report details if available
        if not st.session_state.df_credit_uploaded.empty and 'יתרת חוב' in st.session_state.df_credit_uploaded.columns:
            financial_context += f"  - מתוכם, סך יתרת חוב מדוח אשראי שנותח: {st.session_state.total_debt_from_credit_report if st.session_state.total_debt_from_credit_report is not None else 'לא חושב':,.0f} ₪\n"
            financial_context += "  - פירוט חובות מדוח נתוני אשראי (עיקרי):\n"
            df_credit_cleaned = st.session_state.df_credit_uploaded.copy()
            df_credit_cleaned['יתרת חוב'] = pd.to_numeric(df_credit_cleaned['יתרת חוב'], errors='coerce').fillna(0)
            df_credit_cleaned['יתרה שלא שולמה'] = pd.to_numeric(df_credit_cleaned['יתרה שלא שולמה'], errors='coerce').fillna(0)

            max_credit_entries_to_list = 15 # Increased limit slightly
            for i, row in df_credit_cleaned.head(max_credit_entries_to_list).iterrows():
                 # Ensure row data is valid before formatting
                 סוג_עסקה = row.get('סוג עסקה', 'לא ידוע')
                 שם_בנק = row.get('שם בנק/מקור', 'לא ידוע')
                 יתרת_חוב = row['יתרת חוב'] if pd.notna(row['יתרת חוב']) else 0
                 יתרה_שלא_שולמה = row['יתרה שלא שולמה'] if pd.notna(row['יתרה שלא שולמה']) else 0
                 financial_context += f"    - {סוג_עסקה} ב{שם_בנק}: יתרת חוב {יתרת_חוב:,.0f} ₪ (פיגור: {יתרה_שלא_שולמה:,.0f} ₪)\n"

            if len(df_credit_cleaned) > max_credit_entries_to_list:
                financial_context += f"    ... ועוד {len(df_credit_cleaned) - max_credit_entries_to_list} פריטים בדוח האשראי.\n"
        elif st.session_state.get('uploaded_credit_file_name'): # If file was uploaded but processing failed
             financial_context += "- דוח נתוני אשראי הועלה אך לא ניתן היה לחלץ ממנו נתונים.\n"
        else:
             financial_context += "- לא הועלה דוח נתוני אשראי.\n"


        # Add bank balance trend info if available
        if not st.session_state.df_bank_uploaded.empty:
            financial_context += f"- נותח דוח בנק מסוג: {st.session_state.bank_type_selected}\n"
            df_bank_plot = st.session_state.df_bank_uploaded.dropna(subset=['Date', 'Balance']).sort_values(by='Date').reset_index(drop=True)
            if not df_bank_plot.empty:
                start_date_str = df_bank_plot['Date'].min().strftime('%d/%m/%Y') if not df_bank_plot['Date'].empty and pd.notna(df_bank_plot['Date'].min()) else 'לא ידוע'
                end_date_str = df_bank_plot['Date'].max().strftime('%d/%m/%Y') if not df_bank_plot['Date'].empty and pd.notna(df_bank_plot['Date'].max()) else 'לא ידוע'
                start_balance = df_bank_plot.iloc[0]['Balance'] if not df_bank_plot.empty and pd.notna(df_bank_plot.iloc[0]['Balance']) else np.nan
                end_balance = df_bank_plot.iloc[-1]['Balance'] if not df_bank_plot.empty and pd.notna(df_bank_plot.iloc[-1]['Balance']) else np.nan

                financial_context += f"  - מגמת יתרת חשבון בנק לתקופה מ-{start_date_str} עד {end_date_str}:\n"
                financial_context += f"    - יתרת פתיחה: {start_balance:,.0f} ₪\n" if pd.notna(start_balance) else "    - יתרת פתיחה: לא ידוע\n"
                financial_context += f"    - יתרת סגירה: {end_balance:,.0f} ₪\n" if pd.notna(end_balance) else "    - יתרת סגירה: לא ידוע\n"
                if pd.notna(start_balance) and pd.notna(end_balance):
                     financial_context += f"    - שינוי בתקופה: {(end_balance - start_balance):,.0f} ₪\n"
            else:
                 financial_context += "  - לא ניתן לחלץ נתוני מגמה מדוח הבנק.\n"
        elif st.session_state.bank_type_selected != "ללא דוח בנק": # If bank type was selected but processing failed
             financial_context += f"- דוח בנק מסוג {st.session_state.bank_type_selected} הועלה אך לא ניתן היה לחלץ ממנו נתונים.\n"
        else:
             financial_context += "- לא הועלה דוח בנק.\n"


        financial_context += f"- הכנסה שנתית: {annual_income_ans:,.0f} ₪\n"
        financial_context += f"- יחס חוב להכנסה שנתית: {debt_to_income_ratio_ans:.2%}\n"
        financial_context += f"- סיווג מצב פיננסי ראשוני: {classification} ({description})\n"

        financial_context += "\nתשובות נוספות מהשאלון:\n"

        # Include relevant questionnaire answers, skipping technical keys or ones already summarized
        # Define a dictionary for mapping internal keys to friendly labels
        friendly_key_map = {
            'q1_unusual_event': 'האם קרה משהו חריג שגרם לפנייה',
            'q2_other_funding': 'מקורות מימון אחרים שנבדקו',
            'q3_existing_loans_bool_radio': 'קיימות הלוואות נוספות (ללא משכנתא)',
            'q3_loan_repayment_amount': 'גובה החזר חודשי להלוואות נוספות',
            'q4_financially_balanced_bool_radio': 'מאוזנים כלכלית כרגע',
            'q4_situation_change_next_year': 'שינוי צפוי במצב בשנה הקרובה',
            'arrears_collection_proceedings_radio': 'קיימים פיגורים/הליכי גבייה',
            'can_raise_50_percent_radio': 'יכולת לגייס 50% מהחוב ממקורות תמיכה',
            # Add other keys if needed and not covered above
        }

        for key, value in st.session_state.answers.items():
            # Skip keys that are already explicitly summarized or are internal calculation results
            if key in ['total_net_income', 'total_fixed_expenses', 'monthly_balance', 'total_debt_amount', 'annual_income', 'debt_to_income_ratio',
                       'income_employee', 'income_partner', 'income_other', 'expense_rent_mortgage', 'expense_debt_repayments', 'expense_alimony_other']:
                continue # Skip raw numbers that are summed up

            display_key = friendly_key_map.get(key, key.replace('_', ' ').strip()) # Get friendly name or default

            # Format value based on its type
            if isinstance(value, (int, float)):
                 financial_context += f"- {display_key}: {value:,.0f}\n" # Format numbers
            elif isinstance(value, str) and value.strip() != "":
                 financial_context += f"- {display_key}: {value}\n" # Add non-empty strings
            # Skip None, empty strings, or booleans already covered by radio button logic

        financial_context += "\n--- סוף מידע על המשתמש ---\n"
        # Refined system prompt instructions
        financial_context += "אתה יועץ פיננסי מומחה לכלכלת המשפחה בישראל. המשתמש הזין ו/או העלה נתונים פיננסיים המסוכמים לעיל. ספק ייעוץ פרקטי, ברור, אמפתי ומותאם אישית על בסיס הנתונים שסופקו. ענה בעברית רהוטה. השתמש בסיווג המצב (ירוק/צהוב/אדום) כבסיס להמלצות הראשוניות והרחב עליהן. התייחס לנתונים הספציפיים שסופקו מדוחות או מהשאלון כרלוונטי. אל תמציא נתונים או מקורות מימון שלא צוינו. אם מידע חיוני לשאלה חסר בנתונים שסופקו, ציין זאת. הדגש את סך החובות ויחס החוב להכנסה כנקודות מרכזיות. עזור למשתמש להבין את מצבו ולהתוות צעדים ראשונים אפשריים."


        # Display chat messages from history
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle new user input
        if prompt := st.chat_input("שאל אותי כל שאלה על מצבך הפיננסי או כלכלת המשפחה..."):
            # Add user message to state and display
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Add a temporary assistant placeholder to state immediately
            st.session_state.chat_messages.append({"role": "assistant", "content": ""})
            assistant_message_index = len(st.session_state.chat_messages) - 1

            # Prepare messages for API: system message + all previous messages (excluding the temporary placeholder)
            messages_for_api = [
                {"role": "system", "content": financial_context}
            ] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages[:-1]] # Use history *before* the current assistant turn

            # --- ADD LOGGING HERE ---
            logging.info("Messages sent to OpenAI API:")
            logging.info(messages_for_api)
            # ------------------------

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                try:
                    stream = client.chat.completions.create(
                        model="gpt-4o-mini", # Using a more cost-effective model
                        messages=messages_for_api,
                        stream=True
                    )

                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")

                    message_placeholder.markdown(full_response)

                except APIError as e:
                    logging.error(f"OpenAI API Error (Status Code {e.status_code}): {e.response.text}", exc_info=True)
                    # Check if it's specifically a context length error (status 400, type 'context_length_exceeded')
                    error_detail = "אירעה שגיאה בתקשורת עם שירות הייעוץ הווירטואלי."
                    if e.status_code == 400 and "'code': 'context_length_exceeded'" in str(e.response.text):
                         error_detail = "ההיסטוריה של הצ'אט ופרטי המצב הפיננסי ארוכים מדי. נא ללחוץ על 'התחל מחדש' בסרגל הצד כדי לנקות את הנתונים ולהתחיל שיחה חדשה."
                    else:
                         error_detail += f" (שגיאה: {e.status_code})" # Add status code for other 400s
                    full_response = f"מצטער, {error_detail}"
                    message_placeholder.error(full_response)
                except Exception as e:
                    logging.error(f"An unexpected error occurred during OpenAI API call: {e}", exc_info=True)
                    full_response = "מצטער, אירעה שגיאה בלתי צפויה בעת יצירת התגובה. אנא נסה/י שוב מאוחר יותר."
                    message_placeholder.error(full_response)

                # Update the content of the assistant's message in session state
                st.session_state.chat_messages[assistant_message_index]["content"] = full_response

            # Rerun the app to display the updated chat history
            st.rerun()

    else:
        st.warning("שירות הצ'אט אינו זמין. אנא ודא/י שמפתח ה-API של OpenAI הוגדר כהלכה.")
