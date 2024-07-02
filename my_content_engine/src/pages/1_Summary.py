import streamlit as st 
import json 

# Set page configuration including title and icon
st.set_page_config(page_title="Summary")
# Load summary data from a JSON file
with open("C:/Users/91638/Desktop/SUMMER2024/my_content_engine/data/summaries.json", "r") as f:
    summaries = json.load(f)
    
# Display a markdown header for the report summary
st.markdown("# PDF Summary")
# Iterate over each title and corresponding text in the summaries
for title, text in summaries.items():
    # Display each title as a markdown sub-header
    st.markdown(f"### {title}")
    # Display the text associated with the title
    st.markdown(text)