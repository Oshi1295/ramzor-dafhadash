import fitz  # PyMuPDF
import pandas as pd
import re

def parse_credit_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)

    matches = re.findall(r'(.*?)\n(.*?)\n(.*?)\n(.*?)\n(.*?)\n', text)
    records = []
    for match in matches:
        row = list(match)
        if any("₪" in cell or cell.strip().isdigit() for cell in row):
            records.append(row)

    df = pd.DataFrame(records, columns=["סוג עסקה", "יתרה שלא שולמה", "יתרת חוב", "גובה מסגרת/הלוואה", "מקור"], dtype=str)
    df = df[df["יתרת חוב"].str.replace(",", "").str.strip().str.replace("₪", "").str.isnumeric()]
    df["יתרת חוב"] = df["יתרת חוב"].str.replace(",", "").astype(float)

    total_debt = df["יתרת חוב"].sum()
    summary = {"total_debt": total_debt}
    return df, summary
