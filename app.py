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

# ניתוח קבצים
if credit_file and (files1 or files2):
    st.subheader("🔎 סיכום ניתוח כלכלי")
    all_bank_files = banks1 + banks2
    all_rows = []
    for pdf, bank_name in all_bank_files:
        df, _ = parse_bank_pdf(pdf)
        if not df.empty:
            all_rows.append(df)

    if all_rows:
        bank_df = pd.concat(all_rows)
        bank_df["תאריך"] = pd.to_datetime(bank_df["תאריך"], dayfirst=True, errors='coerce')
        bank_df = bank_df.sort_values("תאריך")

        income_df = bank_df[bank_df.signed_amount > 0]
        expense_df = bank_df[bank_df.signed_amount < 0]
        total_income = income_df.signed_amount.sum()
        total_expense = expense_df.signed_amount.sum()
        net_flow = total_income + total_expense

        st.write(f'**סה"כ הכנסות:** {total_income:,.0f} ש"ח')
        st.write(f'**סה"כ הוצאות:** {-total_expense:,.0f} ש"ח')
        st.write(f'**תזרים חודשי נטו:** {net_flow:,.0f} ש"ח')

        trend = bank_df.groupby("תאריך")["signed_amount"].sum().cumsum()
        fig = px.line(trend, title="📈 תנועת חשבון לאורך זמן", labels={"value": "יתרה מצטברת", "תאריך": "תאריך"})
        st.plotly_chart(fig, use_container_width=True)

        pie1 = px.pie(expense_df, values="signed_amount", names="תיאור", title="פילוח הוצאות")
        pie2 = px.pie(income_df, values="signed_amount", names="תיאור", title="פילוח הכנסות")
        st.plotly_chart(pie1, use_container_width=True)
        st.plotly_chart(pie2, use_container_width=True)

    credit_df, credit_summary = parse_credit_pdf(credit_file)
    total_debt = credit_summary['total_debt']
    yearly_income = income_slider * 12
    ratio = total_debt / yearly_income if yearly_income > 0 else 0
    st.write(f'**סה"כ חוב:** {total_debt:,.0f} ש"ח')
    st.write(f'**יחס חוב להכנסה שנתית:** {ratio:.2f}')

    if ratio < 1:
        color = "🟢 מצב תקין"
    elif ratio < 2:
        color = "🟡 בינוני"
    else:
        color = "🔴 בסיכון"
    st.write(f'**רמזור:** {color}')

    with st.expander("📄 דוח נתוני אשראי"):
        st.dataframe(credit_df)

    st.subheader("📄 הורד סיכום PDF")
    if st.button("📥 הורד דוח מסכם"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, txt="רמזור דף חדש - סיכום", ln=1, align="C")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=f"\nהכנסה: {income_slider}\nהוצאה: {expenses_slider}\nתזרים נטו: {net_flow:,.0f}\nסה\"כ חוב: {total_debt:,.0f}\nיחס חוב/הכנסה: {ratio:.2f}\nרמזור: {color}")
        pdf.cell(200, 10, txt="חתימה: עמותת דף חדש", ln=1, align="R")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf.output(tmp.name)
            with open(tmp.name, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="סיכום_רמזור.pdf">📄 לחץ כאן להורדת הסיכום</a>'
                st.markdown(href, unsafe_allow_html=True)
else:
    st.info("יש להעלות לפחות קובץ אחד של דוח אשראי + קבצי עו\"ש כדי להפיק ניתוח.")
