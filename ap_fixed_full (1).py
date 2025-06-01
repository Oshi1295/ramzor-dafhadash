
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
st.set_page_config(layout="wide", page_title="מומחה כלכלת המשפחה GPT", page_icon="📊")

st.set_page_config(
    layout="wide",
    page_title="מומחה כלכלת המשפחה GPT",
    page_icon="💰"
)

st.title("💰 צ'אטבוט מומחה לכלכלת המשפחה")

# דוגמה לטבלת יתרות עם תאריך בלבד
st.subheader("📊 מגמת יתרת חשבון בנק לאורך זמן")
df = pd.DataFrame({
    "Date": pd.date_range("2024-01-01", periods=10, freq="M"),
    "Balance": [-3000, -2500, -2200, -1800, -1200, -500, 400, 700, 1000, 1200]
})
df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

# גרף בצבע אדום/ירוק
colors = ["red" if x < 0 else "green" for x in df["Balance"]]
fig, ax = plt.subplots()
ax.plot(df["Date"], df["Balance"], marker="o", color="black")
for i in range(len(df)):
    ax.plot(df["Date"][i], df["Balance"][i], marker="o", color=colors[i])
plt.xticks(rotation=45)
st.pyplot(fig)

# צ'אט בוט פשוט עם תשובה מקצועית עד 4 משפטים
st.subheader("💬 צ'אט עם מומחה כלכלת המשפחה")
user_input = st.text_input("מה תרצה לשאול?", "")
if user_input:
    response = "בהתאם לשאלתך, מומלץ לבחון את מצב ההוצאות החודשיות ולוודא שיש יתרה חיובית קבועה. אם יש מינוס כרוני, יש צורך בבחינה מחדש של התקציב. חשוב לבנות רזרבה. שקול להתייעץ עם יועץ פנסיוני מורשה."
    st.markdown(f'<div style="direction: rtl; text-align: right;">{response}</div>', unsafe_allow_html=True)
