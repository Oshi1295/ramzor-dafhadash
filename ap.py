# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import logging
import unicodedata  # For normalization
import re
import io # For BytesIO
import traceback
import numpy as np

# PDF Parsing libraries
import pymupdf as fitz # PyMuPDF, Hapoalim & Credit Report
import pdfplumber # Leumi & Discount

from openai import OpenAI

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- OpenAI Client Setup ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error(f"שגיאה בטעינת מפתח OpenAI: {e}. אנא ודא שהוא מוגדר נכון ב-st.secrets.")
    client = None

# --- Helper Functions (Common for PDF parsers) ---
def clean_number_general(text):
    if text is None: return None
    text = str(text).strip()
    text = re.sub(r'[₪,]', '', text)
    if text.startswith('(') and text.endswith(')'): text = '-' + text[1:-1]
    if text.endswith('-'): text = '-' + text[:-1]
    try: return float(text)
    except ValueError: logging.warning(f"Could not convert '{text}' to float."); return None

def parse_date_general(date_str):
    if date_str is None: return None
    try: return datetime.strptime(date_str.strip(), '%d/%m/%Y')
    except ValueError:
        try: return datetime.strptime(date_str.strip(), '%d/%m/%y')
        except ValueError: logging.warning(f"Could not parse date: {date_str}"); return None

def normalize_text_general(text):
    if text is None: return None
    return unicodedata.normalize('NFC', str(text))

# --- HAPOALIM PARSER ---
def extract_transactions_from_pdf_hapoalim(pdf_content_bytes, filename_for_logging="hapoalim_pdf"):
    transactions = []
    try:
        doc = fitz.open(stream=pdf_content_bytes, filetype="pdf")
    except Exception as e:
        logging.error(f"Hapoalim: Failed to open/process PDF {filename_for_logging}: {e}", exc_info=True)
        return pd.DataFrame()

    date_pattern_end = re.compile(r"(\d{1,2}/\d{1,2}/\d{4})\s*$")
    balance_pattern_start = re.compile(r"^\s*(₪?-?[\d,]+\.\d{2})")

    for page_num, page in enumerate(doc):
        lines = page.get_text("text", sort=True).splitlines()
        for line_num, line_text in enumerate(lines):
            original_line = line_text
            line_normalized = normalize_text_general(line_text.strip())
            if not line_normalized: continue

            date_match = date_pattern_end.search(original_line)
            if date_match:
                date_str = date_match.group(1)
                parsed_date = parse_date_general(date_str)
                if not parsed_date: continue

                balance_match = balance_pattern_start.search(original_line)
                if balance_match:
                    balance_str = balance_match.group(1)
                    balance = clean_number_general(balance_str)
                    if balance is not None:
                        transactions.append({
                            'Date': parsed_date,
                            'Balance': balance,
                            'SourceFile': filename_for_logging,
                            'LineText': original_line.strip() # For context if needed
                        })
    doc.close()
    if not transactions:
        return pd.DataFrame()

    df = pd.DataFrame(transactions)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Balance'] = pd.to_numeric(df['Balance'])
    df = df.sort_values(by=['Date', 'SourceFile', 'LineText'])
    df = df.drop_duplicates(subset='Date', keep='last').reset_index(drop=True)
    return df[['Date', 'Balance']]


# --- LEUMI PARSER ---
def clean_transaction_amount_leumi(text):
    if text is None or pd.isna(text) or text == '': return None
    text = str(text).strip().replace('₪', '').replace(',', '')
    if '.' not in text: return None # Require decimal for amount
    text = text.lstrip('\u200b')
    try:
        if text.count('.') > 1:
            parts = text.split('.')
            text = parts[0] + '.' + "".join(parts[1:])
        val = float(text)
        if abs(val) > 1_000_000: return None # Basic sanity check
        return val
    except ValueError: return None

def clean_number_leumi(text):
    if text is None or pd.isna(text) or text == '': return None
    text = str(text).strip().replace('₪', '').replace(',', '')
    text = text.lstrip('\u200b')
    try:
        if text.count('.') > 1:
            parts = text.split('.')
            text = parts[0] + '.' + "".join(parts[1:])
        return float(text)
    except ValueError: return None

def parse_date_leumi(date_str):
    if date_str is None or pd.isna(date_str) or not isinstance(date_str, str): return None
    date_str = date_str.strip()
    if not date_str: return None
    try: return datetime.strptime(date_str, '%d/%m/%Y').date()
    except ValueError:
        try: return datetime.strptime(date_str, '%d/%m/%y').date()
        except ValueError: return None

def normalize_text_leumi(text):
    if text is None or pd.isna(text): return None
    text = str(text).replace('\r', ' ').replace('\n', ' ')
    text = unicodedata.normalize('NFC', text.strip())
    if any('\u0590' <= char <= '\u05EA' for char in text): # Check for Hebrew characters
       words = text.split()
       reversed_text = ' '.join(words[::-1]) # Reverse word order for Hebrew
       return reversed_text
    return text

def parse_leumi_transaction_line_extracted_order_v2(line_text, previous_balance):
    line = line_text.strip()
    if not line: return None

    pattern = re.compile(
        r"^([\-\u200b\d,\.]+)\s+"           # 1: Balance
        r"(\d{1,3}(?:,\d{3})*\.\d{2})?\s*" # 2: Amount (Optional, specific format)
        r"(\S+)\s+"                        # 3: Reference
        r"(.*?)\s+"                        # 4: Description
        r"(\d{1,2}/\d{1,2}/\d{2,4})\s+"     # 5: Date
        r"(\d{1,2}/\d{1,2}/\d{2,4})$"       # 6: Value Date
    )
    match = pattern.match(line)
    if not match: return None

    balance_str, amount_str, reference_str, description_raw, date_str, value_date_str = match.groups()

    current_balance = clean_number_leumi(balance_str)
    parsed_date = parse_date_leumi(date_str)
    # parsed_value_date = parse_date_leumi(value_date_str) # Not strictly needed for balance trend
    description = normalize_text_leumi(description_raw)

    if parsed_date is None or current_balance is None: return None

    amount = clean_transaction_amount_leumi(amount_str)
    debit = None
    credit = None

    if amount is not None and amount != 0:
        if previous_balance is not None:
            balance_diff = current_balance - previous_balance
            tolerance = 0.02 # Small tolerance for floating point issues
            if abs(balance_diff + amount) < tolerance: debit = amount
            elif abs(balance_diff - amount) < tolerance: credit = amount
            # else: logging.warning(f"Leumi: Amount mismatch. Amt:{amount}, BalDiff:{balance_diff:.2f}, Line: {line_text[:80]}")
    elif amount is None: # No explicit amount, means it's likely a balance line, not a transaction line
        return None

    return {
        'Date': parsed_date, #'ValueDate': parsed_value_date,
        'Description': description, 'Reference': reference_str,
        'Debit': debit, 'Credit': credit, 'Balance': current_balance,
        'OriginalLine': line_text
    }

def extract_leumi_transactions_line_by_line(pdf_content_bytes, filename_for_logging="leumi_pdf"):
    transactions = []
    logging.info(f"\n--- Leumi: Processing {filename_for_logging} using Line-by-Line Regex ---")
    try:
        with pdfplumber.open(io.BytesIO(pdf_content_bytes)) as pdf:
            previous_balance = None
            first_transaction_processed = False

            for page_number, page in enumerate(pdf.pages):
                text = page.extract_text(x_tolerance=2, y_tolerance=2, layout=True)
                if not text: continue
                lines = text.splitlines()
                page_transactions_added = 0

                for line_num, line_text in enumerate(lines):
                    cleaned_line = line_text.strip()
                    if not cleaned_line: continue

                    parsed_transaction_data = parse_leumi_transaction_line_extracted_order_v2(cleaned_line, previous_balance)

                    if parsed_transaction_data:
                        current_balance = parsed_transaction_data['Balance']
                        if not first_transaction_processed:
                            previous_balance = current_balance
                            first_transaction_processed = True
                        else:
                            if parsed_transaction_data['Debit'] is not None or parsed_transaction_data['Credit'] is not None:
                                transactions.append(parsed_transaction_data)
                                previous_balance = current_balance
                                page_transactions_added += 1
                            else: # Parsed structure OK, but Debit/Credit still None
                                previous_balance = current_balance # Update to allow potential recovery
                # logging.info(f"Leumi: Added {page_transactions_added} transactions on page {page_number + 1}.")
    except Exception as e:
        logging.error(f"Leumi: Failed to process PDF file {filename_for_logging}: {e}", exc_info=True)
        return pd.DataFrame()

    if not transactions: return pd.DataFrame()
    
    df_transactions = pd.DataFrame(transactions)
    df_transactions['Date'] = pd.to_datetime(df_transactions['Date'])
    df_transactions['Balance'] = pd.to_numeric(df_transactions['Balance'], errors='coerce')
    df_transactions = df_transactions.sort_values(by=['Date', 'Balance'], ascending=[True, False]).reset_index(drop=True)
    # For Leumi, since we might get multiple entries for the same date (transactions + final balance),
    # we want the last recorded balance for that day.
    df_final = df_transactions.groupby('Date')['Balance'].last().reset_index()
    return df_final[['Date', 'Balance']]


# --- DISCOUNT PARSER ---
def reverse_hebrew_text_discount(text):
    if not text: return text
    words = text.split()
    # For Discount, the specific example seemed to imply word reversal then sentence reversal
    reversed_words = [word[::-1] for word in words] 
    return ' '.join(reversed_words[::-1]) 

def parse_discont_transaction_line(line_text):
    line = line_text.strip()
    if not line: return None

    date_pattern = re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}/\d{1,2}/\d{2,4})$")
    date_match = date_pattern.search(line)
    if not date_match: return None

    date_str2 = date_match.group(1) # Transaction date
    # date_str1 = date_match.group(2) # Value date - not used for balance trend
    parsed_date = parse_date_general(date_str2)
    if not parsed_date: return None

    line_before_dates = line[:date_match.start()].strip()
    balance_amount_pattern = re.compile(r"^([₪\-,\d]+\.\d{2})\s+([₪\-,\d]+\.\d{2})")
    balance_amount_match = balance_amount_pattern.search(line_before_dates)
    if not balance_amount_match: return None

    balance_str = balance_amount_match.group(1)
    # amount_str = balance_amount_match.group(2) # Amount - not used for balance trend
    balance = clean_number_general(balance_str)
    # amount = clean_number_general(amount_str)

    if balance is None: return None
    
    # Description/Ref extraction (simplified as we only need balance for now)
    # ref_desc_text = line_before_dates[balance_amount_match.end():].strip()
    # description = reverse_hebrew_text_discount(ref_desc_text) # Rough

    return {'Date': parsed_date, 'Balance': balance} # Simplified for balance trend

def extract_and_parse_discont_pdf(pdf_content_bytes, filename_for_logging="discount_pdf"):
    transactions = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_content_bytes)) as pdf:
            logging.info(f"Discount: Processing file: {filename_for_logging} with {len(pdf.pages)} pages.")
            for page_number, page in enumerate(pdf.pages):
                text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if text:
                    lines = text.splitlines()
                    for line_text in lines:
                        parsed_transaction = parse_discont_transaction_line(line_text)
                        if parsed_transaction:
                            transactions.append(parsed_transaction)
    except Exception as e:
        logging.error(f"Discount: Failed to open or process PDF file {filename_for_logging}: {e}", exc_info=True)
        return pd.DataFrame()

    if not transactions: return pd.DataFrame()
    
    df_transactions = pd.DataFrame(transactions)
    df_transactions['Date'] = pd.to_datetime(df_transactions['Date'])
    df_transactions = df_transactions.sort_values(by='Date').reset_index(drop=True)
    # Similar to Leumi, take the last balance for each day if multiple entries exist
    df_final = df_transactions.groupby('Date')['Balance'].last().reset_index()
    return df_final[['Date', 'Balance']]


# --- CREDIT REPORT PARSER ---
COLUMN_HEADER_WORDS_CR = {
    "שם", "מקור", "מידע", "מדווח", "מזהה", "עסקה", "מספר", "עסקאות",
    "גובה", "מסגרת", "מסגרות", "סכום", "הלוואות", "מקורי", "יתרת", "חוב",
    "יתרה", "שלא", "שולמה", "במועד"
}
BANK_KEYWORDS_CR = {"בנק", "בע\"מ", "אגוד", "דיסקונט", "לאומי", "הפועלים", "מזרחי",
                 "טפחות", "הבינלאומי", "מרכנתיל", "אוצר", "החייל", "ירושלים",
                 "איגוד", "מימון", "ישיר", "כרטיסי", "אשראי", "מקס", "פיננסים",
                 "כאל", "ישראכרט"}

def process_entry_final_cr(entry_data, section, all_rows_list):
    if not entry_data or not entry_data.get('bank') or len(entry_data.get('numbers', [])) < 2: return

    bank_name_raw = entry_data['bank']
    bank_name_cleaned = re.sub(r'\s*XX-[\w\d\-]+.*', '', bank_name_raw).strip()
    bank_name_cleaned = re.sub(r'\s+\d{1,3}(?:,\d{3})*$', '', bank_name_cleaned).strip()
    bank_name_cleaned = re.sub(r'\s+בע\"מ$', '', bank_name_cleaned).strip()
    bank_name_final = bank_name_cleaned if bank_name_cleaned else bank_name_raw

    is_likely_bank = any(kw in bank_name_final for kw in ["בנק", "לאומי", "הפועלים", "דיסקונט", "מזרחי", "הבינלאומי", "מרכנתיל", "ירושלים", "איגוד"])
    is_non_bank_entity = any(kw in bank_name_final for kw in ["מימון ישיר", "מקס איט", "כרטיסי אשראי", "כאל", "ישראכרט"])

    if is_likely_bank and not bank_name_final.endswith("בע\"מ"):
        bank_name_final += " בע\"מ"
    elif is_non_bank_entity and not bank_name_final.endswith("בע\"מ"):
         if any(kw in bank_name_final for kw in ["מקס איט פיננסים", "מימון ישיר נדל\"ן ומשכנתאות"]):
              bank_name_final += " בע\"מ"

    numbers = entry_data['numbers']
    num_count = len(numbers)
    limit_col, original_col, outstanding_col, unpaid_col = np.nan, np.nan, np.nan, np.nan

    if num_count >= 2:
        val1 = numbers[0]; val2 = numbers[1]; val3 = numbers[2] if num_count >= 3 else 0.0

        if section in ["עו\"ש", "מסגרת אשראי"]:
            limit_col = val1; outstanding_col = val2; unpaid_col = val3
        elif section in ["הלוואה", "משכנתה"]:
            if num_count >= 3:
                 if val1 < 50 and val1 == int(val1) and num_count >= 4: # num_transactions heuristic
                      original_col = numbers[1]; outstanding_col = numbers[2]; unpaid_col = numbers[3]
                 else:
                     original_col = val1; outstanding_col = val2; unpaid_col = val3
            elif num_count == 2: # Fallback for 2 numbers
                 original_col = val1; outstanding_col = val2; unpaid_col = 0.0
        else: # Unknown section type - best guess
            original_col = val1; outstanding_col = val2; unpaid_col = val3

        all_rows_list.append({
            "סוג עסקה": section, "שם בנק/מקור": bank_name_final,
            "גובה מסגרת": limit_col, "סכום מקורי": original_col,
            "יתרת חוב": outstanding_col, "יתרה שלא שולמה": unpaid_col
        })

def extract_credit_data_final_v13(pdf_content_bytes, filename_for_logging="credit_report_pdf"):
    extracted_rows = []
    logging.info(f"\n--- CreditReport: Starting V13 Extraction for: {filename_for_logging} ---")
    try:
        with fitz.open(stream=pdf_content_bytes, filetype="pdf") as doc:
            current_section = None; current_entry = None
            last_line_was_id = False; potential_bank_continuation_candidate = False
            section_patterns = {
                "חשבון עובר ושב": "עו\"ש", "הלוואה": "הלוואה", "משכנתה": "משכנתה",
                "מסגרת אשראי מתחדשת": "מסגרת אשראי",
            }
            number_line_pattern = re.compile(r"^\s*(-?\d{1,3}(?:,\d{3})*\.?\d*)\s*$")

            for page_num, page in enumerate(doc):
                text = page.get_text("text"); lines = text.splitlines()
                for i, line_text in enumerate(lines):
                    original_line = line_text; line = line_text.strip()
                    if not line: potential_bank_continuation_candidate = False; continue

                    is_section_header = False
                    for header_keyword, section_name in section_patterns.items():
                        if header_keyword in line and len(line) < len(header_keyword) + 20 and line.count(' ') < 5:
                            if current_entry and not current_entry.get('processed', False):
                                process_entry_final_cr(current_entry, current_section, extracted_rows)
                            current_section = section_name; current_entry = None
                            last_line_was_id = False; potential_bank_continuation_candidate = False
                            is_section_header = True; break
                    if is_section_header: continue

                    if line.startswith("סה\"כ"):
                        if current_entry and not current_entry.get('processed', False):
                            process_entry_final_cr(current_entry, current_section, extracted_rows)
                        current_entry = None; last_line_was_id = False; potential_bank_continuation_candidate = False
                        continue

                    if current_section:
                        number_match = number_line_pattern.match(line)
                        is_id_line = line.startswith("XX-") and len(line) > 5
                        is_header_word = any(word == line for word in COLUMN_HEADER_WORDS_CR)
                        is_noise_line = is_header_word or line in [':', '.'] or (len(line)<3 and not line.isdigit())

                        if number_match:
                            if current_entry:
                                try:
                                    number = float(number_match.group(1).replace(",", ""))
                                    num_list = current_entry.get('numbers', [])
                                    if last_line_was_id and len(num_list) >= 2:
                                        if not current_entry.get('processed', False):
                                             process_entry_final_cr(current_entry, current_section, extracted_rows)
                                        new_entry = {'bank': current_entry['bank'], 'numbers': [number], 'processed': False}
                                        current_entry = new_entry
                                    else:
                                        if len(num_list) < 4: current_entry['numbers'].append(number)
                                except ValueError: pass
                            last_line_was_id = False; potential_bank_continuation_candidate = False; continue
                        elif is_id_line:
                            last_line_was_id = True; potential_bank_continuation_candidate = False; continue
                        elif is_noise_line:
                            last_line_was_id = False; potential_bank_continuation_candidate = False; continue
                        else:
                             cleaned_line_for_kw_check = re.sub(r'\s*XX-[\w\d\-]+.*', '', line).strip()
                             cleaned_line_for_kw_check = re.sub(r'\d+$', '', cleaned_line_for_kw_check).strip()
                             contains_keyword = any(kw in cleaned_line_for_kw_check for kw in BANK_KEYWORDS_CR)
                             is_potential_bank = contains_keyword or len(cleaned_line_for_kw_check) > 6
                             common_continuations = ["לישראל", "בע\"מ", "ומשכנתאות", "נדל\"ן", "דיסקונט", "הראשון", "פיננסים"]
                             is_continuation = (potential_bank_continuation_candidate and current_entry and
                                                not current_entry.get('numbers') and
                                                any(cleaned_line_for_kw_check.startswith(cont) for cont in common_continuations))

                             if is_continuation:
                                 appendix = cleaned_line_for_kw_check
                                 if appendix:
                                     current_entry['bank'] += " " + appendix
                                     current_entry['bank'] = current_entry['bank'].replace(" בע\"מ בע\"מ", " בע\"מ")
                                 potential_bank_continuation_candidate = True
                             elif is_potential_bank:
                                 if current_entry and not current_entry.get('processed', False):
                                     process_entry_final_cr(current_entry, current_section, extracted_rows)
                                 current_entry = {'bank': line, 'numbers': [], 'processed': False}
                                 potential_bank_continuation_candidate = True
                             else:
                                 potential_bank_continuation_candidate = False
                             last_line_was_id = False
            if current_entry and not current_entry.get('processed', False):
                process_entry_final_cr(current_entry, current_section, extracted_rows)
    except Exception as e:
        logging.error(f"CreditReport: FATAL ERROR processing {filename_for_logging}: {e}", exc_info=True)
        return pd.DataFrame()

    if not extracted_rows: return pd.DataFrame()
    df = pd.DataFrame(extracted_rows)
    final_cols = ["סוג עסקה", "שם בנק/מקור", "גובה מסגרת", "סכום מקורי", "יתרת חוב", "יתרה שלא שולמה"]
    for col in final_cols:
        if col not in df.columns: df[col] = np.nan
    df = df[final_cols]
    # Ensure numeric types for calculation columns
    for col in ["גובה מסגרת", "סכום מקורי", "יתרת חוב", "יתרה שלא שולמה"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="מומחה כלכלת המשפחה GPT", page_icon="💰")
st.title("💰 צ'אטבוט מומחה לכלכלת המשפחה")
st.markdown("העלה את דוחות הבנק ודוח נתוני האשראי שלך, ספק הכנסה חודשית, וקבל ניתוח פיננסי וייעוץ.")

# Initialize session state variables
if 'df_bank' not in st.session_state: st.session_state.df_bank = pd.DataFrame()
if 'df_credit' not in st.session_state: st.session_state.df_credit = pd.DataFrame()
if 'total_debts' not in st.session_state: st.session_state.total_debts = 0
if 'annual_income' not in st.session_state: st.session_state.annual_income = 0
if 'debt_to_income_ratio' not in st.session_state: st.session_state.debt_to_income_ratio = 0
if 'classification' not in st.session_state: st.session_state.classification = None
if 'classification_stage' not in st.session_state: st.session_state.classification_stage = 0 # 0: initial, 1: ratio calculated, 2: q1 asked, 3: q2 asked
if 'collection_proceedings' not in st.session_state: st.session_state.collection_proceedings = None
if 'can_raise_funds' not in st.session_state: st.session_state.can_raise_funds = None
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if "messages" not in st.session_state: st.session_state.messages = []


# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("העלאת נתונים וקלט")
    bank_type_options = ["ללא דוח בנק", "הפועלים", "דיסקונט", "לאומי"]
    selected_bank_type = st.selectbox("בחר סוג דוח בנק:", bank_type_options, key="bank_type_selector")

    uploaded_bank_file = None
    if selected_bank_type != "ללא דוח בנק":
        uploaded_bank_file = st.file_uploader(f"העלה דוח בנק ({selected_bank_type})", type="pdf", key="bank_pdf_uploader")

    uploaded_credit_file = st.file_uploader("העלה דוח נתוני אשראי", type="pdf", key="credit_pdf_uploader")
    monthly_income_input = st.number_input("הכנסה חודשית כוללת של משק הבית (בש\"ח):", min_value=0, value=15000, step=100, key="monthly_income_val")

    if st.button("נתח נתונים", key="analyze_button"):
        st.session_state.analysis_done = False # Reset for new analysis
        st.session_state.classification = None
        st.session_state.classification_stage = 0
        st.session_state.collection_proceedings = None
        st.session_state.can_raise_funds = None
        st.session_state.df_bank = pd.DataFrame() # Reset previous bank data
        st.session_state.df_credit = pd.DataFrame() # Reset previous credit data


        with st.spinner("מעבד נתונים... נא להמתין."):
            # Process Bank File
            if uploaded_bank_file is not None and selected_bank_type != "ללא דוח בנק":
                bank_file_bytes = uploaded_bank_file.getvalue()
                if selected_bank_type == "הפועלים":
                    st.session_state.df_bank = extract_transactions_from_pdf_hapoalim(bank_file_bytes, uploaded_bank_file.name)
                elif selected_bank_type == "לאומי":
                    st.session_state.df_bank = extract_leumi_transactions_line_by_line(bank_file_bytes, uploaded_bank_file.name)
                elif selected_bank_type == "דיסקונט":
                    st.session_state.df_bank = extract_and_parse_discont_pdf(bank_file_bytes, uploaded_bank_file.name)
                
                if st.session_state.df_bank.empty:
                    st.warning(f"לא הצלחנו לחלץ נתונים מדוח הבנק ({selected_bank_type}). אנא בדוק את הקובץ או נסה סוג אחר.")
                else:
                    st.success(f"דוח בנק ({selected_bank_type}) עובד בהצלחה!")

            # Process Credit File
            if uploaded_credit_file is not None:
                credit_file_bytes = uploaded_credit_file.getvalue()
                st.session_state.df_credit = extract_credit_data_final_v13(credit_file_bytes, uploaded_credit_file.name)
                if st.session_state.df_credit.empty:
                    st.warning("לא הצלחנו לחלץ נתונים מדוח האשראי. אנא בדוק את הקובץ.")
                else:
                    st.success("דוח נתוני אשראי עובד בהצלחה!")
                    st.session_state.total_debts = st.session_state.df_credit['יתרת חוב'].sum()
            else:
                st.error("נא להעלות דוח נתוני אשראי לניתוח מלא.")
                st.stop() # Stop processing if no credit report

            st.session_state.annual_income = monthly_income_input * 12
            if st.session_state.annual_income > 0:
                st.session_state.debt_to_income_ratio = st.session_state.total_debts / st.session_state.annual_income
            else:
                st.session_state.debt_to_income_ratio = float('inf') # Or handle as error

            st.session_state.analysis_done = True
            st.session_state.classification_stage = 1 # Move to classification
            st.rerun() # Rerun to update main page layout with new data and classification steps


# --- Main Area for Results & Chat ---
if st.session_state.analysis_done:
    st.header("📊 סיכום פיננסי")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 סך חובות כולל", f"{st.session_state.total_debts:,.0f} ₪")
    col2.metric("📈 הכנסה שנתית", f"{st.session_state.annual_income:,.0f} ₪")
    col3.metric("⚖️ יחס חוב להכנסה שנתית", f"{st.session_state.debt_to_income_ratio:.2%}")

    # --- Classification Logic ---
    if st.session_state.classification_stage == 1: # Ratio calculated, start classification
        ratio = st.session_state.debt_to_income_ratio
        if ratio < 1:
            st.session_state.classification = "ירוק"
            st.session_state.classification_stage = 4 # Done
        elif 1 <= ratio <= 2:
            st.session_state.classification_stage = 2 # Ask Q1
        else: # ratio > 2
            st.session_state.classification = "אדום"
            st.session_state.classification_stage = 4 # Done
        st.rerun() # Rerun to show question or final classification

    if st.session_state.classification_stage == 2: # Ask Q1
        st.subheader("שאלת הבהרה לסיווג:")
        q1_answer = st.radio("האם נפתחו נגדך הליכי גבייה?", ("כן", "לא"), index=None, key="q1_collection") # index=None for no default
        if q1_answer == "כן":
            st.session_state.collection_proceedings = True
            st.session_state.classification = "אדום"
            st.session_state.classification_stage = 4 # Done
            st.rerun()
        elif q1_answer == "לא":
            st.session_state.collection_proceedings = False
            st.session_state.classification_stage = 3 # Ask Q2
            st.rerun()
            
    if st.session_state.classification_stage == 3: # Ask Q2 (only if Q1 was "לא")
        st.subheader("שאלת הבהרה נוספת לסיווג:")
        q2_answer = st.radio(f"סך החובות שלך הוא {st.session_state.total_debts:,.0f} ₪. האם אתה מסוגל לגייס {st.session_state.total_debts * 0.5:,.0f} ₪ (50% מהחוב) תוך זמן סביר?", 
                             ("כן", "לא"), index=None, key="q2_raise_funds") # index=None for no default
        if q2_answer == "כן":
            st.session_state.can_raise_funds = True
            st.session_state.classification = "צהוב"
            st.session_state.classification_stage = 4 # Done
            st.rerun()
        elif q2_answer == "לא":
            st.session_state.can_raise_funds = False
            st.session_state.classification = "אדום"
            st.session_state.classification_stage = 4 # Done
            st.rerun()

    if st.session_state.classification_stage == 4 and st.session_state.classification:
        classification_text = st.session_state.classification
        if classification_text == "ירוק":
            st.success(f"🟢 סיווג: {classification_text}! מצב פיננסי נראה תקין ביחס להכנסה.")
        elif classification_text == "צהוב":
            st.warning(f"🟡 סיווג: {classification_text}. יש לשים לב ולשקול פעולות לשיפור המצב.")
        elif classification_text == "אדום":
            st.error(f"🔴 סיווג: {classification_text}. מומלץ מאוד לפעול לשיפור המצב הפיננסי ו/או לפנות לייעוץ מקצועי.")

    # --- Visualizations ---
    st.header("🎨 ויזואליזציות")
    
    # Visualization 1: Debt Breakdown (Pie Chart)
    if not st.session_state.df_credit.empty and 'סוג עסקה' in st.session_state.df_credit.columns and 'יתרת חוב' in st.session_state.df_credit.columns:
        st.subheader("חלוקת החובות לפי סוג")
        # Group smaller categories into 'אחר' if there are many types
        debt_summary = st.session_state.df_credit.groupby("סוג עסקה")["יתרת חוב"].sum().reset_index()
        debt_summary = debt_summary[debt_summary['יתרת חוב'] > 0] # Only positive debts
        if not debt_summary.empty:
            fig_debt_pie = px.pie(debt_summary, values='יתרת חוב', names='סוג עסקה', 
                                  title='פירוט יתרות חוב לפי סוג עסקה',
                                  color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_debt_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_debt_pie, use_container_width=True)
        else:
            st.info("אין נתוני חובות להצגה בתרשים עוגה.")
    
    # Visualization 2: Debt vs. Income (Bar Chart)
    st.subheader("השוואת סך חובות להכנסה שנתית")
    if st.session_state.total_debts > 0 and st.session_state.annual_income > 0 :
        comparison_data = pd.DataFrame({
            'קטגוריה': ['סך חובות', 'הכנסה שנתית'],
            'סכום בש"ח': [st.session_state.total_debts, st.session_state.annual_income]
        })
        fig_debt_income_bar = px.bar(comparison_data, x='קטגוריה', y='סכום בש"ח', 
                                     title='השוואת סך חובות להכנסה שנתית',
                                     color='קטגוריה', text_auto=True,
                                     labels={'סכום בש"ח': 'סכום בש"ח'})
        fig_debt_income_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_debt_income_bar, use_container_width=True)

    # Visualization 3: Bank Balance Trend (Line Chart)
    if not st.session_state.df_bank.empty and 'Date' in st.session_state.df_bank.columns and 'Balance' in st.session_state.df_bank.columns:
        st.subheader("מגמת יתרת חשבון בנק לאורך זמן")
        df_bank_plot = st.session_state.df_bank.dropna(subset=['Date', 'Balance'])
        if not df_bank_plot.empty:
            fig_balance_trend = px.line(df_bank_plot, x='Date', y='Balance', 
                                        title=f'מגמת יתרת חשבון ({selected_bank_type})', markers=True,
                                        labels={'Date': 'תאריך', 'Balance': 'יתרה בש"ח'})
            fig_balance_trend.update_layout(xaxis_title='תאריך', yaxis_title='יתרה בש"ח')
            st.plotly_chart(fig_balance_trend, use_container_width=True)
        else:
            st.info("אין מספיק נתונים תקינים להצגת גרף מגמת יתרת חשבון.")

    # Display DataFrames (optional, can be in expanders)
    with st.expander(" הצג טבלת נתוני אשראי מפורטת"):
        if not st.session_state.df_credit.empty:
            st.dataframe(st.session_state.df_credit.style.format({
                "גובה מסגרת": '{:,.0f}', "סכום מקורי": '{:,.0f}',
                "יתרת חוב": '{:,.0f}', "יתרה שלא שולמה": '{:,.0f}'
            }))
        else:
            st.write("לא נטענו נתוני אשראי.")

    if selected_bank_type != "ללא דוח בנק":
        with st.expander(f"הצג טבלת יתרות בנק ({selected_bank_type})"):
            if not st.session_state.df_bank.empty:
                st.dataframe(st.session_state.df_bank.style.format({"Balance": '{:,.2f}'}))
            else:
                st.write("לא נטענו נתוני בנק.")

# --- Chatbot Interface ---
st.header("💬 צ'אט עם מומחה כלכלת המשפחה")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if client: # Only show chat input if OpenAI client is initialized
    if prompt := st.chat_input("שאל אותי כל שאלה על מצבך הפיננסי או כלכלת המשפחה..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Constructing context for the chatbot
            financial_context = ""
            if st.session_state.analysis_done:
                financial_context += f"\n\n--- סיכום פיננסי של המשתמש ---\n"
                financial_context += f"סך חובות: {st.session_state.total_debts:,.0f} ₪\n"
                financial_context += f"הכנסה שנתית: {st.session_state.annual_income:,.0f} ₪\n"
                financial_context += f"יחס חוב להכנסה: {st.session_state.debt_to_income_ratio:.2%}\n"
                if st.session_state.classification:
                    financial_context += f"סיווג המצב: {st.session_state.classification}\n"
                if st.session_state.classification == "צהוב" or st.session_state.classification == "אדום":
                    if st.session_state.collection_proceedings is not None:
                        financial_context += f"הליכי גבייה: {'כן' if st.session_state.collection_proceedings else 'לא'}\n"
                    if st.session_state.can_raise_funds is not None:
                        financial_context += f"יכולת לגייס 50% מהחוב: {'כן' if st.session_state.can_raise_funds else 'לא'}\n"
                financial_context += "--- סוף סיכום פיננסי ---\n"


            messages_for_api = [
                {"role": "system", "content": f"אתה מומחה לכלכלת המשפחה בישראל. המטרה שלך היא לספק ייעוץ פיננסי ברור, מעשי וקל להבנה. ענה בעברית. {financial_context}"}
            ] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            
            try:
                stream = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages_for_api,
                    stream=True,
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"אני מצטער, התרחשה שגיאה בעת יצירת התשובה: {e}"
                message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.warning("שירות הצ'אט אינו זמין עקב בעיה בטעינת מפתח OpenAI.")

st.sidebar.markdown("---")
st.sidebar.info("פותח ככלי עזר. המידע כאן אינו מהווה ייעוץ פיננסי מקצועי ויש להתייעץ עם גורם מוסמך.")