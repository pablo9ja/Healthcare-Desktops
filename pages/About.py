import streamlit as st

def main():
    st.title(" :bar_chart: Helpman Healthcare About me Interactive Dashboard")

    st.write("Information about the app.")




def about_page():
    def first_page():
        st.title("About Me")
        st.write(
            """
            Hi there! I'm Gemini, a large language model created by Google AI. I can generate text, translate languages, 
            write different kinds of creative content, and answer your questions in an informative way.

            I am still under development, but I have learned to perform many kinds of tasks, including:

            * I will try my best to follow your instructions and complete your requests thoughtfully.
            * I will use my knowledge to answer your questions in a comprehensive and informative way, 
              even if they are open-ended, challenging, or strange.
            * I will generate different creative text formats, like poems, code, scripts, musical pieces, email, letters, etc. 
              I will try my best to fulfill all your requirements.

            I am excited to learn more and keep improving my abilities!
            """
        )

    # --- HERO SECTION ---
    col1, col2 = st.columns([1, 2], gap="small")
    with col1:
        pass

    with col2:
        st.title("Sven Bosau")
        st.write("Senior Data Analyst, assisting enterprises by supporting data-driven decision-making.")

    # --- EXPERIENCE & QUALIFICATIONS ---
    st.write("\n")
    st.subheader("Experience & Qualifications")
    st.write(
        """
        - 7 Years experience extracting actionable insights from data
        - Strong hands-on experience and knowledge in Python and Excel
        - Good understanding of statistical principles and their respective applications
        - Excellent team-player and displaying a strong sense of initiative on tasks
        """
    )

    # --- SKILLS ---
    st.write("\n")
    st.subheader("Hard Skills")
    st.write(
        """
        - Programming: Python (Scikit-learn, Pandas), SQL, VBA
        - Data Visualization: PowerBi, MS Excel, Plotly
        - Modeling: Logistic regression, linear regression, decision trees
        - Databases: Postgres, MongoDB, MySQL
        """
    )

if __name__ == "__main__":
    about_page()
