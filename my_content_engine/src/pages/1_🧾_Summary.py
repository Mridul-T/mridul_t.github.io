import streamlit as st 
import json 

# Set page configuration including title and icon
st.set_page_config(page_title="Summary", page_icon='🧾')
# Load summary data from a JSON file

summary='''

## Overview

I developed a Content Engine , focusing on analyzing and comparing multiple PDF documents. This system leverages Retrieval Augmented Generation (RAG) techniques to efficiently retrieve, assess, and generate insights from the documents. The project involved creating a scalable and modular architecture that ensures data privacy by using local instances of models and embeddings.

## Setup

### Backend Framework
- **LangChain**: I chose LangChain asI'm familiar with it and its a powerful toolkit tailored for building LLM applications with a strong emphasis on retrieval-augmented generation.

### Frontend Framework
- **Streamlit**: Utilized Streamlit to build an interactive web application, providing a user-friendly interface for the Content Engine.

### Vector Store
- **ChromaDB**: Selected ChromaDB to manage and query the embeddings effectively.

### Embedding Model
- **Local Embedding Model**: Downloaded and Implemented a locally running embedding model, `all-MiniLM-L6-v2` to generate vectors from the PDF content, ensuring no external service or API exposure. I chose this model because it's one of the best among small/low dimensional embedding models and it can easily run locally with mediocre hardware.

### Local Language Model (LLM)
- **Local LLM**: Downloaded and Integrated a local instance of a `LLala2 7B Layla` in GGUF format to make it compatible with the llama.cpp and hence can be easily used locally on mediocre hardware, maintaining complete data privacy. 

## Working

1. **Parsing Documents**: Extracted text and structured data from three PDF documents containing the Form 10-K filings of Alphabet Inc., Tesla, Inc., and Uber Technologies, Inc.
2. **Generating Vectors**: Used the local embedding model to create embeddings for the content of these documents.
3. **Storing in Vector Store**: Persisted the generated vectors in ChromaDB for efficient querying.
4. **Configuring Query Engine**: Set up retrieval tasks based on document embeddings to facilitate comparison and insights generation.
5. **Integrating LLM**: Deployed a local instance of a `LLala2 7B Layla` to provide contextual insights based on the retrieved data.
6. **Developing Chatbot Interface**: Built a chatbot interface using Streamlit, enabling users to interact with the system, obtain insights, and compare information across the documents.

This Content Engine effectively analyzes and highlights differences in the provided documents, allowing users to query and retrieve meaningful insights seamlessly. The details of how to run and deployment of this bot on your device locally are given in the readme file in the github repo.

`NOTE`: The bot can work slow depending on the specification of your system.
'''
    
# Display a markdown header for the report summary
st.markdown("# PDF Bot Summary")

st.markdown(summary)