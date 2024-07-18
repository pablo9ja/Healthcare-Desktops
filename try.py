import streamlit as st
from streamlit_option_menu import option_menu
from about_me import about_page
# --- PAGE SETUP ---
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

def chat_bot():
    st.title("Chat Bot")
    st.write("This is the Chat Bot page.")

# --- NAVIGATION SETUP ---
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",  # required
        options=["About Me", "healthcare_dashboard", "Chat Bot"],  # required
        
        default_index=0,  # optional
    )

# --- DISPLAY SELECTED PAGE ---
if selected == "About Me":
    about_page()
  
elif selected == "Chat Bot":
    chat_bot()

# --- SHARED ON ALL PAGES ---

st.sidebar.text("Made with streamlit")
