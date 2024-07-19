import streamlit as st
from pathlib import Path
import importlib.util

# Apply custom CSS
def apply_custom_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Main application
def main():
    st.sidebar.title("My Dashboard Pages")
    apply_custom_css()
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Home", "About", "Finance", "Patient"])

    page_files = {
        "Home": "pages/Home.py",
        "About": "pages/About.py",
        "Finance": "pages/Pinance.py",
        "Patient": "pages/Patient.py"
    }

    if page in page_files:
        page_module = load_module(page, page_files[page])
        page_module.main()

if __name__ == "__main__":
    main()
