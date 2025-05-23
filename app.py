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

st.header("📝 שאלון ראשוני")
income = st.number_input("סך ההכנסה החודשית (₪)", min_value=0, step=500)
income_slider = st.slider("בחר הכנסה", 0, 150000, income, step=500)
if income_slider != income:
    income = income_slider

expenses = st.number_input("סך ההוצאות הקבועות (₪)", min_value=0, step=500)
expenses_slider = st.slider("בחר הוצאה", 0, 150000, expenses, step=500)
if expenses_slider != expenses:
    expenses = expenses_slider

event = st.text_input("האם קרה משהו חריג?")
alt_funding = st.text_input("האם יש מקורות מימון נוספים?")
is_balanced = st.radio("האם אתם מאוזנים כלכלית?", ["כן", "לא"])
will_change = st.radio("צפי לשינוי בשנה הקרובה?", ["כן", "לא"])

st.markdown("---")

st.header("📤 העלאת קבצים")

st.subheader("⬆️ לקוח/ה 1 - עו\"ש")
files1 = st.file_uploader('קבצי עו\"ש', type="pdf", accept_multiple_files=True, key="bank1")
banks1 = [st.selectbox(f"בחר בנק עבור {f.name}", ["הפועלים", "לאומי", "דיסקונט", "מזרחי טפחות", "בינלאומי", "מרכנתיל", "יהב", "אוצר החייל"], key=f.name) for f in files1]

credit1 = st.file_uploader("דוח אשראי לקוח/ה 1", type="pdf", key="credit1")

st.subheader("⬆️ לקוח/ה 2 - עו\"ש")
files2 = st.file_uploader('קבצי עו\"ש', type="pdf", accept_multiple_files=True, key="bank2")
banks2 = [st.selectbox(f"בחר בנק עבור {f.name}", ["הפועלים", "לאומי", "דיסקונט", "מזרחי טפחות", "בינלאומי", "מרכנתיל", "יהב", "אוצר החייל"], key=f"b_{f.name}") for f in files2]

credit2 = st.file_uploader("דוח אשראי לקוח/ה 2", type="pdf", key="credit2")

st.markdown("---")

if (files1 or files2) and (credit1 or credit2):
    st.header("🔎 ניתוח נתונים")
    all_rows = []
    for f in files1 + files2:
        df, _ = parse_bank_pdf(f)
        if not df.empty:
            df["תאריך"] = pd.to_datetime(df["תאריך"], dayfirst=True, errors="coerce")
            all_rows.append(df)
    if all_rows:
        bank_df = pd.concat(all_rows)
        bank_df = bank_df.sort_values("תאריך")
        income_df = bank_df[bank_df.signed_amount > 0]
        expense_df = bank_df[bank_df.signed_amount < 0]
        total_income = income_df.signed_amount.sum()
        total_expense = expense_df.signed_amount.sum()
        net = total_income + total_expense
        st.write(f'**סה\"כ הכנסות:** {total_income:,.0f} ₪')
        st.write(f'**סה\"כ הוצאות:** {-total_expense:,.0f} ₪')
        st.write(f'**תזרים נטו:** {net:,.0f} ₪')

        trend = bank_df.groupby("תאריך")["signed_amount"].sum().cumsum()
        st.plotly_chart(px.line(trend, title="📈 תנועת חשבון לאורך זמן"))

        st.plotly_chart(px.pie(income_df, names="תיאור", values="signed_amount", title="פילוח הכנסות"))
        st.plotly_chart(px.pie(expense_df, names="תיאור", values="signed_amount", title="פילוח הוצאות"))

    # אשראי
    all_credit = [credit1, credit2]
    total_debt = 0
    credit_details = []
    for c in all_credit:
        if c:
            df, summary = parse_credit_pdf(c)
            total_debt += summary["total_debt"]
            credit_details.append(df)
    if credit_details:
        credit_full = pd.concat(credit_details)
        st.dataframe(credit_full)

    yearly_income = income * 12
    ratio = total_debt / yearly_income if yearly_income > 0 else 0
    st.write(f'**סה\"כ חוב:** {total_debt:,.0f} ₪')
    st.write(f'**יחס חוב/הכנסה:** {ratio:.2f}')

    if ratio < 1:
        color = "🟢 תקין"
    elif ratio < 2:
        color = "🟡 בינוני"
    else:
        color = "🔴 סיכון גבוה"
    st.subheader(f"סיכום רמזור: {color}")

    # PDF
    st.subheader("📄 הורדת סיכום PDF")
    if st.button("📥 צור קובץ PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, txt="סיכום רמזור דף חדש", ln=1, align="C")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"סה\"כ חוב: {total_debt:,.0f} ₪\nיחס חוב/הכנסה: {ratio:.2f}\nתזרים נטו: {net:,.0f} ₪\nרמזור: {color}")
        pdf.cell(200, 10, txt="חתימה: עמותת דף חדש", ln=1, align="R")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf.output(tmp.name)
            with open(tmp.name, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="סיכום_רמזור.pdf">📄 הורד את הדוח</a>'
                st.markdown(href, unsafe_allow_html=True)

st.markdown("---")
st.header("📞 יצירת קשר")
st.markdown("""
**זקוקים לסיוע נוסף?**
ניתן לפנות לעמותת דף חדש ונחזור אליכם בהקדם.

- טלפון: 050-0000000  
- מייל: info@dafhadash.org.il  
- אתר: www.dafhadash.org.il
""")
