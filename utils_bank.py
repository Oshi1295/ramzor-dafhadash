import fitz  # PyMuPDF
import pandas as pd
import re

def parse_bank_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)

    pattern = r'(\d{2}/\d{2}/\d{4}).*?([\u0590-\u05FF\s"\']+)([\-–]?\d[\d,\.]*₪)'
    matches = re.findall(pattern, text)

    rows = []
    for date, desc, amount in matches:
        try:
            clean_amount = float(re.sub(r'[₪,]', '', amount))
            signed = -clean_amount if '-' in amount or '–' in amount else clean_amount
            rows.append({"תאריך": date, "תיאור": desc.strip(), "signed_amount": signed})
        except:
            continue

    df = pd.DataFrame(rows)
    return df, {}
