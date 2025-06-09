import streamlit as st
from datetime import datetime, timedelta
import requests
import os

# --- Helper functions ---
def get_day_suffix(day):
    if 11 <= day <= 13:
        return 'th'
    return {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')

def build_parliament_pdf_url(date_obj):
    day = date_obj.day
    month = date_obj.strftime("%B")
    year = date_obj.year
    suffix = get_day_suffix(day)
    formatted_date = f"{day}{suffix} {month}, {year}"
    encoded_date = formatted_date.replace(" ", "%20").replace(",", "%2C")
    url = f"https://www.parliament.gh/epanel/docs/pb/{encoded_date}.pdf"
    return url, formatted_date

def download_parliament_pdfs(start_date, end_date, save_folder="proceedings"):
    os.makedirs(save_folder, exist_ok=True)

    current_date = start_date
    downloaded = []

    while current_date <= end_date:
        url, display_name = build_parliament_pdf_url(current_date)
        filename = os.path.join(save_folder, f"{display_name}.pdf")

        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            downloaded.append(f"{display_name}.pdf")
        current_date += timedelta(days=1)

    return downloaded

# --- Streamlit UI ---
# Sidebar
st.sidebar.caption(':orange[This app was built on debate reports from the Parliament of Ghana. Credit: github.com/agbozo1]')

col1, _, col3 = st.columns([3, 1, 2])
with col1:
    st.subheader('📥 Download 🇬🇭 Parliamentary Briefs')
with col3:
    st.image('imgs/Ghana_Parliament_Emblem.png', width=150)

st.write("Select a date range to download available parliamentary debate reports.")
st.info("Files will be saved to the `proceedings/` folder and replaced if they already exist.")

col1, col2 = st.columns(2)
with col1:
    start = st.date_input("Start Date", datetime.today() - timedelta(days=30))
with col2:
    end = st.date_input("End Date", datetime.today())

if st.button("Download Reports"):
    if start > end:
        st.error("❌ Start date must be before end date.")
    else:
        with st.spinner("🔄 Downloading and saving files..."):
            files = download_parliament_pdfs(start, end)
        
        if files:
            st.success(f"✅ Downloaded {len(files)} document(s).")
            st.write("📄 Saved Files:")
            for f in files:
                st.write(f"• {f}")
        else:
            st.warning("⚠️ No files were found in the selected date range.")
