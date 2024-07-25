import streamlit as st
#import os
#import warnings
#warnings.filterwarnings('ignore')
from streamlit_option_menu import option_menu
import Revenue_streams, Patient,Hospital_Performance, Hospital_Staff,doctor, Quality_of_care

st.set_page_config(page_title="Healthcare!!!", page_icon=":bar_chart:", layout="wide")

class MultiApp:
    def __init__(self):
        self.apps=[]
    def add_app(self, title, function):
        self.apps.append({"title":title, "function":function})

    def run():
        # app = st.sidebar(
        with st.sidebar:        
            app = option_menu(
                menu_title='Pondering ',
                options=['Revenue_streams','Patient','Hospital_Performance','doctor','Hospital_Staff','Quality_of_care'],
                icons=['house-fill','person-circle','trophy-fill','chat-fill','info-circle-fill'],
                menu_icon='chat-text-fill',
                default_index=1,
                styles={
                    "container": {"padding": "5!important","background-color":'black'},
        "icon": {"color": "white", "font-size": "23px"}, 
        "nav-link": {"color":"white","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
        "nav-link-selected": {"background-color": "#02ab21"},}
                
                )
        if app == "Revenue_streams":
            Revenue_streams.app()
        if app == "Patient":
            Patient.app()    
        if app == "Hospital_Performance":
            Hospital_Performance.app()        
        if app == 'Hospital_Staff':
            Hospital_Staff.app()
        if app == 'doctor':
            doctor.app()    
        if app=='Quality_of_care':
            Quality_of_care.app()    
                
    run()      