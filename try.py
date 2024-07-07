import streamlit as st
import openai
import plotly.express as px
import pandas as pd
from datetime import datetime

# Configuration
openai.api_key = 'sk-proj-1tgzl3ay7B6ijHHOxDIET3BlbkFJ3qkPbNXOZ4FPrwFowq7L'

# Initialize session state for conversation and statistics
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'stats' not in st.session_state:
    st.session_state.stats = []

# Function to handle user input and generate responses
def generate_response(user_input):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"User: {user_input}\nAI:",
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = response.choices[0].text.strip()
    return message

# Streamlit UI
st.title("Healthcare Chatbot")

# Chatbot interaction
st.subheader("Ask a healthcare-related question:")
user_input = st.text_input("Your question:")

if user_input:
    response = generate_response(user_input)
    st.session_state.conversation.append({"timestamp": datetime.now(), "user": user_input, "bot": response})
    st.session_state.stats.append({"timestamp": datetime.now(), "query": user_input, "response": response})
    st.text_area("Chatbot Response:", value=response, height=200, max_chars=None, key=None)

# Display conversation history
st.subheader("Conversation History")
for entry in st.session_state.conversation:
    st.write(f"{entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"**User:** {entry['user']}")
    st.write(f"**Bot:** {entry['bot']}")
    st.write("---")

# Real-time Dashboard
st.subheader("Real-time Dashboard")

# Convert stats to DataFrame for visualization
df = pd.DataFrame(st.session_state.stats)

# Display number of queries over time
if not df.empty:
    df['hour'] = df['timestamp'].dt.hour
    query_counts = df.groupby('hour').size().reset_index(name='counts')
    fig = px.line(query_counts, x='hour', y='counts', title='Number of Queries Over Time')
    st.plotly_chart(fig)

    # Display common topics (simplified example using keyword matching)
    df['topic'] = df['query'].apply(lambda x: 'symptom' if 'symptom' in x else 'general')
    topic_counts = df['topic'].value_counts().reset_index()
    topic_counts.columns = ['topic', 'count']
    fig = px.bar(topic_counts, x='topic', y='count', title='Common Topics')
    st.plotly_chart(fig)

# User feedback
st.subheader("User Feedback")
feedback = st.radio("Was this information helpful?", ('Yes', 'No'))
if feedback:
    st.write("Thank you for your feedback!")
