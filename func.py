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

# Load custom CSS
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
uploaded_file = st.sidebar.file_uploader("fake_healthcare.csv", type="csv")

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
    #st.write(filtered_df)  # Example of displaying filtered data; adjust as needed

# Footer
st.sidebar.text("Made with Streamlit")



# Provide plot functions
def plot_daily_admissions_vs_visits_donut_chart(df):
    # Calculate the sum of daily visits and daily admissions
    total_visits = df['daily_visits'].sum()
    total_admissions = df['daily_admissions'].sum()

    # Data for the donut chart
    labels = ['Total Number of Daily Visits', 'Total Number of Daily Admissions']
    sizes = [total_visits, total_admissions]
    colors = ['#FFD700', '#FFDAB9']  # Yellow and peach colors
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.3))
    plt.title('Distribution of Daily Admissions per Visit')
    plt.show()
    st.pyplot(plt)

def plot_equipment_donut_chart(df):
    total_revenue = df['equip_count'].sum()
    total_profit = df['equip_use'].sum()
    labels = ['Total Number of Equipment', 'Total Number of Equipment Used']
    sizes = [total_revenue, total_profit]
    colors = ['#4CAF50', '#FFDAB9']  # Green and peach colors
    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.3))
    plt.title('Distribution of Equipment Usage')
    plt.show()
    st.pyplot(plt)

def plot_donut_chart(data_frame): 
    total_values = [data_frame[col].sum() for col in ['daily_revenue', 'daily_profit']]

    # Plotting the donut chart
    plt.figure(figsize=(5, 5))
    plt.pie(total_values, labels=['Daily Revenue', 'Daily Profit'], colors= ['#4CAF50', '#2196F3'], autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.3))
    plt.title('Distribution of Daily Revenue and Daily Profit')
    plt.show()
    st.pyplot(plt)

def plot_pie_chart(d, value_column, group_column):
   
    average_employee_count = df.groupby('departments')['employee_count'].mean()
    plt.figure(figsize=(6, 6))
    plt.pie(average_employee_count, labels=average_employee_count.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Average Employee Count in Each Department')
    plt.show()
    st.pyplot(plt)    

def plot_employee_attendance(df):
    if 'date' not in df.index:
        # Create a new 'date' column based on a date range or other datetime data
        start_date = '2024-01-01'
        end_date = '2024-04-10'
        date_range = pd.date_range(start=start_date, end=end_date, periods=len(df))  # Adjust periods as needed

        df['date'] = date_range

    # Ensure 'date' column is correctly set as datetime
    df['date'] = pd.to_datetime(df['date'])

    # Extract day of the week (0 = Monday, 6 = Sunday)
    df['day_of_week'] = df['date'].dt.dayofweek

    # Calculate total employee count and resigned employees count of the week
    daily_attendance = df.groupby('day_of_week')[['employee_count', 'employee_resign']].sum()

    # Calculate percentage of resignations relative to total employee count
    daily_attendance['resignation_percentage'] = (daily_attendance['employee_resign'] / daily_attendance['employee_count']) * 100

    # Labels for days of the week
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Plotting total employee count and resigned employees count of the week
    plt.figure(figsize=(12, 6))

    # Width of the bars
    bar_width = 0.35

    # Position of bars on X-axis
    r1 = range(len(days))
    r2 = [x + bar_width for x in r1]

    bars1 = plt.bar(r1, daily_attendance['employee_count'], color='skyblue', width=bar_width, edgecolor='grey', label='Employee Count')
    bars2 = plt.bar(r2, daily_attendance['employee_resign'], color='lightcoral', width=bar_width, edgecolor='grey', label='Resigned Employees')

    plt.xlabel('Day of the Week')
    plt.ylabel('Count')
    plt.title('Employee Count and Resignations by Day of the Week')
    plt.xticks([r + bar_width / 2 for r in range(len(days))], days)
    plt.legend()

    # Adding percentages as annotations
    for bar1, bar2 in zip(bars1, bars2):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        idx = bars1.index(bar1)
        plt.text(bar1.get_x() + bar1.get_width() / 2, height1, f'{int(height1)}', ha='center', va='bottom', fontsize=10)
        plt.text(bar2.get_x() + bar2.get_width() / 2, height2, f'{int(height2)} ({daily_attendance.loc[idx]["resignation_percentage"]:.1f}%)', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

def plot_visits_per_day_of_week(df):   
    if 'date' not in df.index:
        # Create a new 'date' column based on a date range or other datetime data
        start_date = '2024-01-01'
        end_date = '2024-04-10'
        date_range = pd.date_range(start=start_date, end=end_date, periods=len(df)) 
        df['date'] = date_range
        df['date'] = pd.to_datetime(df['date'])

    if 'date' in df.columns:
        # Extract day of the week (0 = Monday, 6 = Sunday)
        df['day_of_week'] = df['date'].dt.dayofweek

        # Calculate total visits per day of the week
        visits_per_day = df.groupby('day_of_week')['daily_visits'].sum()

        # Labels for days of the week
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        # Plotting pie chart
        plt.figure(figsize=(5, 5))
        plt.pie(visits_per_day, labels=days, autopct='%1.1f%%', startangle=140)
        plt.title('Distribution of Visits per Day of the Week')
        plt.show()
        st.pyplot(plt)
    else:
        print("Error: 'date' column not found in DataFrame. Please check your data.")
    

def plot_average_wait_time(df):
    agg_data = df.groupby('departments').agg({
        'wait_time': 'sum',  # Sum wait time in timedelta format (or numeric)
        'daily_visits': 'sum'
    }).reset_index()

    # Calculate the average wait time (ensure numeric types)
    if pd.api.types.is_datetime64_dtype(agg_data['wait_time']):
        # Handle datetime format (assuming represents total time)
        agg_data['Average_wait_time'] = agg_data['wait_time'].dt.total_seconds() / agg_data['daily_visits']
    else:
        # Handle numeric wait time (assuming represents wait time in minutes)
        agg_data['Average_wait_time'] = agg_data['wait_time'] / agg_data['daily_visits']

    # Sort by average wait time (descending)
    agg_data = agg_data.sort_values(by='Average_wait_time', ascending=False)

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(agg_data['departments'], agg_data['Average_wait_time'], color='lime')
    plt.xlabel('Departments')
    plt.ylabel('Average Wait Time (minutes)')
    plt.title('Average Wait Time per Department')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

def plot_average_length_of_stays(df):   
    # Calculate the average length of stay
    df['average_length_of_stay'] = df['patient_days'] / df['daily_discharge']
    # Aggregate the data to get average length of stay per department
    agg_data = df.groupby('departments')['average_length_of_stay'].mean().reset_index()
    # Sort the data by 'average_length_of_stay' in ascending order
    agg_data = agg_data.sort_values(by='average_length_of_stay')
    plt.figure(figsize=(10, 6))
    plt.bar(agg_data['departments'], agg_data['average_length_of_stay'], color='green')
    plt.xlabel('Department')
    plt.ylabel('Average Length of Stay (days)')
    plt.title('Average Length of Stay per Department')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

def plot_average_length_of_stay(df):
    df['average_length_of_stay'] = df['patient_days'] / df['daily_discharge']
    # Aggregate the data to get average length of stay per department
    agg_data = df.groupby('departments')['average_length_of_stay'].mean().reset_index()
    # Sort the data by 'average_length_of_stay' in ascending order
    agg_data = agg_data.sort_values(by='average_length_of_stay')

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(agg_data['departments'], agg_data['average_length_of_stay'], color='green')
    plt.xlabel('Department')
    plt.ylabel('Average Length of Stay (days)')
    plt.title('Average Length of Stay per Department')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

def plot_average_length_of_stay(df):
    agg_data = df.groupby('departments').sum().reset_index()
    agg_data['Average_Length_of_Stay'] = agg_data['patient_days'] / agg_data['daily_discharge']
    agg_data = agg_data.sort_values(by='Average_Length_of_Stay')
    plt.figure(figsize=(10, 6))
    plt.bar(agg_data['departments'], agg_data['Average_Length_of_Stay'], color='olive')
    plt.xlabel('Departments')
    plt.ylabel('Average Length of Stay (days)')
    plt.title('Average Length of Stay per Department')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

def plot_total_daily_visits_by_department(df):   
    department_visits = df.groupby('departments')['daily_visits'].sum()
    department_visits = department_visits.sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    department_visits.plot(kind='barh', title='Total Daily Visits by Department')
    plt.ylabel('Department')
    plt.xlabel('Total Daily Visits')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)


def plot_daily_visits_and_admissions(df):
    df_sorted = df.groupby('departments').sum().sort_values(by='daily_visits', ascending=False)
    width = 0.4
    x = np.arange(len(df_sorted))
    plt.figure(figsize=(14, 8))
    plt.bar(x - width/2, df_sorted['daily_visits'], width, label='Daily Visits', color='blue')

    # Bar for daily admissions
    plt.bar(x + width/2, df_sorted['daily_admissions'], width, label='Daily Admissions', color='orange')
    # Set the x-axis labels to the sorted department names
    plt.xticks(x, df_sorted.index, rotation=45)
    plt.xlabel('Departments')
    plt.ylabel('Count')
    plt.title('Daily Visits and Admissions by Department (Sorted by Daily Visits)')
    plt.legend()
    # Show the plot
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

def plot_total_daily_profit_by_department(df):   
    department_profits = df.groupby('departments')['daily_profit'].sum().sort_values(ascending=False)

    # Plot total daily profit for each department
    plt.figure(figsize=(12, 6))
    department_profits.plot(kind='bar', color='green')

    plt.xlabel('Departments')
    plt.ylabel('Total Daily Profit')
    plt.title('Total Daily Profit by Department')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)


def plot_admissions_discharges_readmissions(df):
    # Group by departments and sum daily admissions, daily discharges, and daily readmissions
    df_grouped = df.groupby(['departments'])[['daily_admissions', 'daily_discharge', 'daily_readmission']].sum()
    df_grouped = df_grouped.sort_values(by='daily_admissions', ascending=True)
    plt.figure(figsize=(10, 8))  # Adjust height as needed for many departments
    ax = df_grouped.plot(kind='barh', stacked=False, color=['blue', 'red', 'green'], ax=plt.gca()) 
    plt.title('Daily Admissions, Discharge, and Readmission by Department')
    plt.xlabel('Count')
    plt.ylabel('Department')

    # Grid and legend
    plt.legend(title='Metric', labels=['Daily Admissions', 'Daily Discharge', 'Daily Readmission'])  # Adjust legend title and labels if needed

    # Show plot
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)



def plot_admissions_and_discharges(df):
    # Group by departments and sum daily admissions and daily discharges
    df_grouped = df.groupby(['departments'])[['daily_admissions', 'daily_discharge']].sum()

    # Sort the data by daily admissions for better readability
    df_grouped = df_grouped.sort_values(by='daily_admissions', ascending=True)

    # Plot
    plt.figure(figsize=(10, 8))  # Adjust height as needed for many departments
    ax = df_grouped.plot(kind='barh', stacked=False, color=['blue', 'red'], ax=plt.gca())  # Use horizontal bar plot

    # Title and labels
    plt.title('Daily Admissions and Discharge by Department')
    plt.xlabel('Count')
    plt.ylabel('Department')
    # Grid and legend
    plt.legend(title='Metric', labels=['Daily Admissions', 'Daily Discharge'])  # Adjust legend title and labels if needed

    # Show plot
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

def plot_beds_in_use_doughnut(df):
    # Sum the total beds and beds in use across all dates
    total_beds_in_use = df['beds_in_use'].sum()
    total_beds = df['total_beds'].sum()

    # Data for the doughnut plot
    sizes = [total_beds_in_use, total_beds - total_beds_in_use]
    labels = ['Beds in Use', 'Available Beds']
    colors = ['#ff9999','#66b3ff']

    # Create a doughnut plot
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, startangle=90, counterclock=False,
            wedgeprops=dict(width=0.3), autopct='%1.1f%%')

    # Add a circle at the center to make it a doughnut plot
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Beds in Use vs Total Beds')
    plt.show()
    st.pyplot(plt)

def plot_revenue_vs_profit_spider_chart(df):
    sorted_df = df.sort_values(by='daily_revenue', ascending=False).tail()
    departments = sorted_df['departments'].tolist()
    revenue = sorted_df['daily_revenue'].tolist()
    profit = sorted_df['daily_profit'].tolist()
    angles = [n / float(len(departments)) * 2 * np.pi for n in range(len(departments))]
    angles += angles[:1]  # Complete the loop
    revenue += revenue[:1]
    profit += profit[:1]
    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, revenue, 'b-', linewidth=2, label='Revenue')
    ax.fill(angles, revenue, 'b', alpha=0.1)
    ax.plot(angles, profit, 'g-', linewidth=2, label='Profit')
    ax.fill(angles, profit, 'g', alpha=0.1)
    plt.xticks(angles[:-1], departments, fontsize=12, ha='right', alpha=0.7)
    plt.title('Revenue vs. Profit by Department', size=20, color='darkblue', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)

def plot_revenue_vs_profit_spider_chart(df, top_n=5):
    sorted_df = df.sort_values(by='daily_revenue', ascending=False).head(top_n)
    departments = sorted_df['departments'].tolist()
    revenue = sorted_df['daily_revenue'].tolist()
    profit = sorted_df['daily_profit'].tolist()
    angles = [n / float(len(departments)) * 2 * np.pi for n in range(len(departments))]
    angles += angles[:1]  # Complete the loop
    revenue += revenue[:1]
    profit += profit[:1]
    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, revenue, 'b-', linewidth=2, label='Revenue')
    ax.fill(angles, revenue, 'b', alpha=0.1)
    ax.plot(angles, profit, 'g-', linewidth=2, label='Profit')
    ax.fill(angles, profit, 'g', alpha=0.1)
    plt.xticks(angles[:-1], departments, fontsize=12, ha='right', alpha=0.7)
    plt.title('Revenue vs. Profit by Department', size=20, color='darkblue', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=12)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

def plot_wordcloud(df, column_name='refer_reason', additional_stopwords=None):
    all_text = " ".join(df[column_name].tolist())
    stopwords = set(STOPWORDS)  # Import stopwords from WordCloud
    if additional_stopwords:
        stopwords.update(additional_stopwords)  # Add additional stopwords if desired
    wordcloud = WordCloud(width=800, height=600, background_color="white", stopwords=stopwords).generate(all_text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="nearest")
    plt.axis("off")
    plt.title("Frequent Words in " + column_name.replace('_', ' ').title())
    plt.show()
    st.pyplot(plt)


def plot_admissions_readmissions(df):
    department_admissions = df.groupby('departments')['daily_admissions'].sum()
    department_readmissions = df.groupby('departments')['daily_readmission'].sum()
    sorted_departments = department_admissions.sort_values(ascending=False).index
    bar_width = 0.35
    index = np.arange(len(sorted_departments))
    plt.figure(figsize=(12, 6))
    plt.bar(index, department_admissions[sorted_departments], bar_width, label='Daily Admissions', color='blue')
    plt.bar(index + bar_width, department_readmissions[sorted_departments], bar_width, label='Daily Readmissions', color='orange')
    plt.xlabel('Departments')
    plt.ylabel('Count')
    plt.title('Daily Admissions and Readmissions by Department')
    plt.xticks(index + bar_width / 2, sorted_departments, rotation=45)
    plt.legend()
    plt.tight_layout()  
    st.pyplot(plt)


def plot_staff_patient_ratio_pie_chart():
    ratio_counts = df['staff_patient_ratio'].value_counts()
    sizes = ratio_counts.values
    labels = ratio_counts.index
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    explode = (0, 0, 0, 0)

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(aspect="equal"))
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                      autopct='%1.1f%%', startangle=140, pctdistance=0.85)
    centre_circle = plt.Circle((0, 0), 0.50, fc='white')
    fig.gca().add_artist(centre_circle)
    ax.axis('equal')
    plt.title('Distribution of Staff to Patient Ratios', pad=20)
    plt.setp(autotexts, size=10, weight="bold", color="white")
    plt.setp(texts, size=12, weight="bold")
    for w in wedges:
        w.set_edgecolor('black')
    plt.tight_layout()
    st.pyplot(fig)

def plot_top_refer_reasons_by_department():
    df['refer_reason'] = df['refer_reason'].str.lower()
    department_df = df[df['departments'] == selected_dep]
    top_refer_reasons_department = department_df['refer_reason'].value_counts().head(16)

    fig, ax = plt.subplots(figsize=(12, 6))
    top_refer_reasons_department.plot(kind='barh', rot=0, color='blue', ax=ax)
    ax.set_xlabel('Counts')
    ax.set_ylabel('Refer Reason')
    ax.set_title(f'Refer Reasons for {selected_dep.capitalize()} Department')
    plt.tight_layout()
    st.pyplot(fig)

def plot_admission_rate():
    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust figure size for landscape
    sns.lineplot(x='date', y='admission_rate', data=filtered_df, ax=ax, marker='o', color='salmon')
    ax.set_title('Admission Rate', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Admission Rate', fontsize=14)
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_daily_revenue():
    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust figure size for landscape
    sns.barplot(x='date', y='daily_revenue', data=filtered_df, ax=ax, color='lightgreen')
    ax.set_title('Daily Revenue', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Revenue ($)', fontsize=14)
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_daily_profit():
    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust figure size for landscape
    sns.barplot(x='date', y='daily_profit', data=filtered_df, ax=ax, color='lightcoral')
    ax.set_title('Daily Profit', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Profit ($)', fontsize=14)
    plt.xticks(rotation=45)
    st.pyplot(fig)