# רמזור דף חדש - אפליקציה מלאה
import streamlit as st
import pandas as pd
import plotly.express as px
from utils_bank import parse_bank_pdf
from utils_credit import parse_credit_pdf
from fpdf import FPDF
import tempfile
import base64
import os

st.set_page_config(page_title="רמזור דף חדש", layout="wide")
st.markdown("""
<style>
body, .stTextInput, .stNumberInput, .stSelectbox, .stRadio, .stFileUploader, .stButton, .stMarkdown, .stDataFrameBlock, .stDataFrame, .stTable {
    direction: rtl;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

st.image("https://i.ibb.co/X5qq2mL/logo.png", width=100)
st.markdown("""
## נכנסתם לחובות? אל פחד!
### נצא מזה ביחד!
יחד נעזור לכם לסגור את החובות ולפתוח דף חדש.
---
""")

# שאלון רמזור בסיסי
st.header("📝 שאלון ראשוני")
event = st.text_input("האם קרה משהו חריג שבגללו פנית?")
alt_funding = st.text_input("האם יש מקורות מימון נוספים שנבדקו?")
income_slider = st.slider("הכנסה חודשית משק הבית (₪)", 0, 150000, 8000, step=500)
expenses_slider = st.slider("הוצאה חודשית קבועה (₪)", 0, 150000, 7000, step=500)
other_loans = st.text_input("האם קיימות הלוואות נוספות? פרט/י והוסף/י גובה החזר חודשי")
is_balanced = st.radio("האם אתם מאוזנים כלכלית?", ["כן", "לא"])
will_change = st.radio("האם צפוי שינוי במצב הכלכלי בשנה הקרובה?", ["כן", "לא"])

st.markdown("---")

# העלאת קבצים לפי בני זוג
st.header("📤 העלאת קבצים - בן/בת זוג")
st.subheader("⬆️ לקוח/ה 1")
files1 = st.file_uploader('העלה קבצי עו"ש', type="pdf", accept_multiple_files=True, key="bank1")
banks1 = []
for f in files1:
    bank_name = st.selectbox(f"בחר את הבנק עבור {f.name}", ["בנק הפועלים", "בנק לאומי", "בנק דיסקונט", "מזרחי טפחות", "הבנק הבינלאומי", "מרכנתיל", "יהב", "אוצר החייל"], key=f.name)
    banks1.append((f, bank_name))

st.subheader("⬆️ לקוח/ה 2")
files2 = st.file_uploader('העלה קבצי עו"ש', type="pdf", accept_multiple_files=True, key="bank2")
banks2 = []
for f in files2:
    bank_name = st.selectbox(f"בחר את הבנק עבור {f.name}", ["בנק הפועלים", "בנק לאומי", "בנק דיסקונט", "מזרחי טפחות", "הבנק הבינלאומי", "מרכנתיל", "יהב", "אוצר החייל"], key=f"b_{f.name}")
    banks2.append((f, bank_name))

credit_file = st.file_uploader("העלה דוח נתוני אשראי (PDF)", type="pdf", key="credit")

st.markdown("---")
