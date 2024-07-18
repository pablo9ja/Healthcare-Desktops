import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from about_me import about_page  # Assuming about_page is defined in about_me module
from bed import plot_beds_in_use, plot_doughnut_for_department, plot_refer_reasons
from func import *

# st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Dictionary mapping departments to colors
department_colors = {
    "emergency": "#FFCCCC",  # Light Red
    "internal_med": "#CCCCFF",  # Light Blue
    "surgery": "#CCFFCC",  # Light Green
    "pediatric": "#FFFFCC",  # Light Yellow
    "obgyn": "#FFCCFF",  # Light Pink
    "cardio": "#CCCCCC",  # Light Gray
    "orthopedic": "#FF9999",  # Light Red
    "neurology": "#66B3FF",  # Light Blue
    "oncology": "#99FF99",  # Light Green
    "radiology": "#FFCC99",  # Light Orange
    "pathology": "#FFD700",  # Light Yellow
    "anesthesiology": "#E6E6FA",  # Light Lavender
    "icu": "#F0E68C",  # Light Khaki
    "psychiatry": "#FFB6C1",  # Light Pink
    "physical_therapy": "#98FB98",  # Pale Green
    "resp_therapy": "#AFEEEE",  # Pale Turquoise
    "nutrition_diet": "#FFDAB9",  # Peach Puff
    "pharmacy": "#FFC0CB",  # Pink
    "laboratory": "#B0E0E6",  # Powder Blue
    "infection_control": "#FFE4B5",  # Moccasin
    "medical_records": "#FFD700",  # Gold
    "admin": "#DDA0DD",  # Plum
    "security": "#FF4500",  # Orange Red
}

st.sidebar.header("Healthcare CSV data")
uploaded_file = st.sidebar.file_uploader("file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("fake_healthcare.csv")  # Default dataset

# Function to set sidebar background color based on department
def set_sidebar_color(department):
    color = department_colors.get(department, "#FFFFFF")  # Default to white if department not found
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{
            background-color: {color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Sidebar navigation setup
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",  # required
        options=["About Me", "Healthcare Dashboard"],  # required
        default_index=0,  # optional
    )

    # Additional page selection for departments (only for Healthcare Dashboard)
    if selected == "Healthcare Dashboard":
        departments = ["Overview"] + list(department_colors.keys())
        page = st.selectbox("Page", departments)

# Display selected page content
if selected == "About Me":
    about_page()
else:
    if page != "Overview":
        selected_dep = page
    else:
        selected_dep = st.sidebar.selectbox("Department", list(department_colors.keys()))

    set_sidebar_color(selected_dep)
    filtered_df = df[df['departments'] == selected_dep]  # Filter dataset based on selected department
    
    # Display department-specific content or visualizations
    st.subheader(f"{selected_dep.replace('_', ' ').title()} Department Overview")
    st.write(filtered_df)  # Example of displaying filtered data; adjust as needed
    

# Footer
st.sidebar.text("Made with Streamlit")




# Display plots based on the selected page
if page == "Overview":
    st.write("### Overview")
    col1, col2,col3= st.columns(3)
    with col1:
        plot_staff_patient_ratio_pie_chart()
    with col2:
        plot_admissions_readmissions(df)
    with col3:
        plot_admissions_discharges_readmissions(df)
    
    col4, col5,col6= st.columns(3)
    with col4:
        plot_wordcloud(df, column_name='refer_reason', additional_stopwords=None)
    with col5:
        plot_average_length_of_stay(df)
    # You can add more plots to these columns as needed
elif page == "emergency":
    st.write("### emergency")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_average_length_of_stay(df)
    with col2:
        plot_doughnut_for_department(df, 'emergency')
    with col3:
        plot_beds_in_use(df, 'emergency')

    col4, col5, col6 = st.columns(3)
    with col4:
        plot_average_length_of_stay(df)

elif page == "internal_med":
    st.write("### internal_med")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_average_length_of_stay(df)
    with col2:
        plot_beds_in_use(df, 'internal_med')
    with col3:
        plot_doughnut_for_department(df, 'emergency')
    col4, col5, col6 = st.columns(3)
    with col4:
        plot_average_length_of_stay(df)

elif page == "surgery":
    st.write("### Surgery")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_average_length_of_stays(df)
    with col2:
        plot_beds_in_use(df, 'surgery')
    with col3:
        plot_doughnut_for_department(df, 'surgery')

    col4, col5, col6 = st.columns(3)
    with col4:
        plot_average_length_of_stays(df)

elif page == "pediatric":
    st.write("### pediatric")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_pie_chart(df, 'employee_count', 'departments')
    with col2:
        plot_beds_in_use(df, 'pediatric')
    with col3:
        plot_pie_chart(df, 'employee_count', 'departments')

    col4, col5, col6 = st.columns(3)
    with col4:
        plot_pie_chart(df, 'employee_count', 'departments')

elif page == "obgyn":
    st.write("### obgyn")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_daily_admissions_vs_visits_donut_chart(df)
    with col2:
        plot_beds_in_use(df, 'obgyn')
    with col3:
        plot_daily_admissions_vs_visits_donut_chart(df)

    col4, col5, col6 = st.columns(3)
    with col4:
        plot_daily_admissions_vs_visits_donut_chart(df)
    

elif page == "cardio":
    st.write("### cardio")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_admission_rate()
    with col2:
        plot_beds_in_use(df, 'cardio')
    with col3:
        plot_daily_revenue()

    col4, col5, col6 = st.columns(3)
    with col4:
        plot_daily_profit()  

elif page == "orthopedic":
    st.write("### orthopedic")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_beds_in_use(df, 'orthopedic')
    with col2:
        plot_admission_rate()
    with col3:
        plot_daily_revenue()

    col4, col5, col6 = st.columns(3)
    with col4:
        plot_daily_profit() 

elif page == "neurology":
    st.write("### neurology")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_beds_in_use(df, 'neurology')
    with col2:
        plot_admission_rate()
    with col3:
        plot_daily_revenue()

    col4, col5, col6 = st.columns(3)
    with col4:
        plot_daily_profit()   

elif page == "oncology":
    st.write("### oncology")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_beds_in_use(df, 'oncology')
    with col2:
        plot_refer_reasons(df, 'oncology')
    with col3:
        plot_daily_revenue()

    col4, col5, col6 = st.columns(3)
    with col4:
        plot_daily_profit()   

elif page == "radiology":
    st.write("### radiology")
    col1, col2, col3 = st.columns(3)
    with col1:
       plot_beds_in_use(df, 'radiology')
    with col2:
        plot_admission_rate()
    with col3:
        plot_daily_revenue()

    col4, col5, col6 = st.columns(3)
    with col4:
        plot_daily_profit()   

elif page == "pathology":
    st.write("### pathology")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_beds_in_use(df, 'pathology')
    with col2:
        plot_refer_reasons(df, 'pathology')
    with col3:
        plot_daily_revenue()

    col4, col5, col6 = st.columns(3)
    with col4:
        plot_daily_profit()   

elif page == "anesthesiology":
    st.write("### anesthesiology")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_beds_in_use(df, 'anesthesiology')
    with col2:
        plot_admission_rate()
    with col3:
        plot_daily_revenue()

    col4, col5, col6 = st.columns(3)
    with col4:
        plot_daily_profit()       

elif page == "icu":
    st.write("### icu")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_beds_in_use(df, 'icu')
    with col2:
        plot_admission_rate()
    with col3:
        plot_daily_revenue()

    col4, col5, col6 = st.columns(3)
    with col4:
        plot_daily_profit()


elif page == "psychiatry":
    st.write("### psychiatry")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_beds_in_use(df, 'psychiatry')
    with col2:
        plot_admission_rate()
    with col3:
        plot_daily_revenue()

    col4, col5, col6 = st.columns(3)
    with col4:
        plot_daily_profit()   

elif page == "physical_therapy":
    st.write("### physical_therapy")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_beds_in_use(df, 'physical_therapy')
    with col2:
        plot_employee_attendance(df)
    with col3:
        plot_employee_attendance(df)

    col4, col5, col6 = st.columns(3)
    with col4:
        plot_employee_attendance(df)

elif page == "resp_therapy":
    st.write("### resp_therapy")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_beds_in_use(df, 'resp_therapy')
    with col2:
        plot_admission_rate()
    with col3:
        plot_daily_revenue()

    col4, col5, col6 = st.columns(3)
    with col4:
        plot_daily_profit() 

elif page == "nutrition_diet":
    st.write("### nutrition_diet")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_beds_in_use(df, 'nutrition_diet')
    with col2:
        plot_admission_rate()
    with col3:
        plot_daily_revenue()

    col4, col5, col6 = st.columns(3)
    with col4:
        plot_daily_profit() 

elif page == "pharmacy":
    st.write("### pharmacy")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_beds_in_use(df, 'pharmacy')
    with col2:
        plot_equipment_donut_chart(df)
    with col3:
        plot_equipment_donut_chart(df)

    col4, col5, col6 = st.columns(3)
    with col4:
        plot_equipment_donut_chart(df)

elif page == "laboratory":
    st.write("### laboratory")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_beds_in_use(df, 'laboratory')
    with col2:
        plot_donut_chart(df)
    with col3:
        plot_donut_chart(df)

    col4, col5, col6 = st.columns(3)
    with col4:
        plot_donut_chart(df)

elif page == "infection_control":
    st.write("### infection_control")
    col1, col2, col3 = st.columns(3)
    with col1:
       plot_admissions_and_discharges(df)
    with col2:
        plot_beds_in_use(df, 'infection_control')
    with col3:
        plot_visits_per_day_of_week(df)
    col4, col5, col6 = st.columns(3)
    with col4:
        plot_visits_per_day_of_week(df)

elif page == "medical_records":
    st.write("### medical_records")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_beds_in_use(df, 'medical_records')
    with col2:
        plot_daily_visits_and_admissions(df)
    with col3:
        plot_daily_visits_and_admissions(df)

    col4, col5, col6 = st.columns(3)
    with col4:
        plot_total_daily_visits_by_department(df)


elif page == "admin":
    st.write("### admin")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_beds_in_use(df, 'admin')
    with col2:
        plot_revenue_vs_profit_spider_chart(df)
    with col3:
        plot_beds_in_use_doughnut(df)

    col4, col5, col6 = st.columns(3)
    with col4:
       plot_total_daily_profit_by_department(df)    

elif page == "security":
    st.write("### security")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_beds_in_use(df, 'security')
    with col2:
        plot_average_wait_time(df)
    with col3:
        plot_average_wait_time(df)

    col4, col5, col6 = st.columns(3)
    with col4:
        plot_average_wait_time(df)    

else:
    st.write(f"### {page.capitalize()} Department Data")
   