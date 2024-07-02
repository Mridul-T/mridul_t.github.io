## Overview

I developed a Content Engine ChatBot, focusing on analyzing and comparing multiple PDF documents. This system leverages Retrieval Augmented Generation (RAG) techniques to efficiently retrieve, assess, and generate insights from the documents. The project involved creating a scalable and modular architecture that ensures data privacy by using local instances of models and embeddings.

## Setup

### Backend Framework
- **LangChain**: I chose LangChain as I'm familiar with it and its a powerful toolkit tailored for building LLM applications with a strong emphasis on retrieval-augmented generation.

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

### Prerequisites
While there are not specific pre requisites to use this code, you need to make aure that your C++ make tools and Windows SDK is up to date.

### Installation
```css
# First create a directory for your project. You can clone this repo or make one on your one. The structure should look like this
my_content_engine/
├── .gitignore
├── data/
│   ├── alphabet_10k.pdf
│   ├── tesla_10k.pdf
│   └── uber_10k.pdf
├── myenv/
├── src/
│   ├── pages/
│   ├── hello.py
│   ├── process_and_store.py
│   ├── utils.py
├── requirements.txt
└── README.md

```

```sh
# Create a virtual python enviornment.
python -m venv myenv

# Activate the enviornment
 myenv\Scripts\activate.bat
# Install dependencies
pip install -r requirements.txt
```
### Download and install the embedding model and the llm either from github or huggingface. 
```sh
# github code to download the models
git lfs install
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

git clone https://huggingface.co/TheBloke/llama2-7B-layla-GGUF/llama2-7b-layla.Q5_K_S.gguf
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/TheBloke/llama2-7B-layla-GGUF/llama2-7b-layla.Q5_K_S.gguf
```
You can use any other model of your choice as well, just make sure to make the required changes in the `utils.py`.

### Once you have downloaded and clone the repository perfectly, we can store the data of the PDFs as embeddings in the vector database. We'll be using `chromadb` for this. Go through the file `process_and_store.py` for more details.
```sh
# You can run this file using
python3 process_and_store.py
```
Make sure you have activated the virtual envirment and installed all the required packages.

### Now all that's left is to host the ChatBot Locally using streamlit and the following command
```sh
streamlit run .\src\Hello.py
```
If you want the details of the implementation of the chatbot, query engine and the vector database. You can go through the files in the `src` folder.

### Example Usage:
![image](https://github.com/Mridul-T/mridul_t.github.io/assets/121259465/da7c6167-7be7-4ca9-abf5-37dd5cc746c2)

`NOTE`: The bot can work slow depending on the specification of your system.
