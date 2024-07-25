import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
df = pd.read_csv("fake_healthcare.csv")


def plot_beds_in_use(df, department):
    # Filter the DataFrame for the specified department
    department_df = df[df['departments'] == department]
    
    if department_df.empty:
        print(f"No data available for the department: {department}")
        return
    
    # Aggregate beds_in_use by day_of_week
    agg_df = department_df.groupby('day_of_week').agg({'beds_in_use': 'sum', 'total_beds': 'first'}).reset_index()
    agg_df['beds_available'] = agg_df['total_beds'] - agg_df['beds_in_use']
    
    # Check for NaN or invalid values
    if agg_df.isna().any().any():
        print("NaN values found in the aggregated data.")
        print(agg_df)
        return

    if (agg_df['beds_in_use'] < 0).any() or (agg_df['total_beds'] <= 0).any():
        print("Invalid bed counts found in the aggregated data.")
        print(agg_df)
        return

    # Define warm colors
    colors = ['#FF6347', '#FFA07A', '#FF7F50', '#FF4500', '#FF8C00', '#FFD700', '#FFA500']

    # Plotting
    plt.figure(figsize=(5, 5))
    plt.pie(agg_df['beds_in_use'], labels=agg_df['day_of_week'], autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops={'edgecolor': 'black'})

    # Draw a circle to make it a doughnut plot
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Display beds available in the center
    plt.text(0, 0, f'Beds Available:\n{agg_df["beds_available"].sum()}',
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=12, weight='bold')

    plt.title(f'Beds in Use by Day of Week - {department}')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
    st.pyplot(plt)

# Call the function to generate the plot for the 'admin' department
plot_beds_in_use(df, 'emergency')



def plot_doughnut_for_department(df, target_department):
    # Check if the target department exists in the DataFrame
    if target_department not in df['departments'].unique():
        print(f"Department '{target_department}' not found in the DataFrame.")
        return

    # Filter the DataFrame for the target department
    department_df = df[df['departments'] == target_department].copy()

    # Handle NaN values by filling them with zeros
    department_df.loc[:, 'beds_in_use'] = department_df['beds_in_use'].fillna(0)
    department_df.loc[:, 'total_beds'] = department_df['total_beds'].fillna(0)

    # Sum the total beds and beds in use across all dates for the current department
    total_beds_in_use = department_df['beds_in_use'].sum()
    total_beds = department_df['total_beds'].sum()

    if total_beds == 0:
        print(f"No bed data available for the '{target_department}' department.")
        return

    # Data for the doughnut plot
    sizes = [total_beds_in_use, total_beds - total_beds_in_use]
    labels = ['Beds in Use', 'Available Beds']
    colors = ['#ff9999', '#66b3ff']

    # Create a doughnut plot
    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, colors=colors, startangle=90, counterclock=False, wedgeprops=dict(width=0.3),
            autopct='%1.1f%%')

    # Add a circle at the center to make it a doughnut plot
    centre_circle = plt.Circle((0, 0), 0.50, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(f'Beds in Use vs Total Beds in {target_department.capitalize()} Department')
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)


# Example usage:
# Assuming 'df' is your DataFrame
# df = pd.read_csv('your_data.csv')  # Load your DataFrame here
plot_doughnut_for_department(df, 'surgery')



def plot_refer_reasons(df, target_department):
   
    # Check if DataFrame is empty
    if df.empty:
        print("DataFrame is empty. Please load data into 'df'.")
        return

    # Ensure 'refer_reason' column is lowercase
    df['refer_reason'] = df['refer_reason'].str.lower()

    # Check if the target department exists in the DataFrame
    if target_department not in df['departments'].unique():
        print(f"Department '{target_department}' not found in the DataFrame.")
        return

    # Filter the DataFrame for the target department
    department_df = df[df['departments'] == target_department]

    # Check if filtered DataFrame is empty
    if department_df.empty:
        print(f"No data found for department '{target_department}'.")
        return

    # Get the top 16 refer reasons and their counts for the target department
    top_refer_reasons_department = department_df['refer_reason'].value_counts().head(16)

    # Check if there are any refer reasons to plot
    if top_refer_reasons_department.empty:
        print(f"No refer reasons found for department '{target_department}'.")
        return

    # Plot the data
    plt.figure(figsize=(10, 5))
    top_refer_reasons_department.plot(kind='barh', rot=0, color='blue')
    plt.xlabel('Counts')
    plt.ylabel('Refer Reason')
    plt.title(f'Refer Reasons for {target_department.capitalize()} Department')
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

# Example usage:
# Assuming 'df' is your DataFrame
# df = pd.read_csv('your_data.csv')  # Load your DataFrame here

# Plot the refer reasons for the 'security' department
plot_refer_reasons(df, 'surgery')
