import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from transformers import AutoModelForCausalLM,AutoTokenizer
import time 

# Initialize SentenceTransformer model for embeddings
embedding_function = SentenceTransformerEmbeddings(model_name="D:/mridul_t.github.io/my_content_engine/all-MiniLM-L6-v2")
# Initialize Chroma vector store for document embeddings
# and create retriever from the vector store
vectordb = Chroma(persist_directory="data/chroma_db", embedding_function=embedding_function)
retriever = vectordb.as_retriever()

# Initialize llama-7b language model with 
model_id = "./llama2-7b-layla.Q5_K_S.gguf"
# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer and don't start making up random questions and answers. Keep the answer as concise as possible. Always say "\nthanks for asking! BBYE!!" at the end of the answer. 
{context}
Question: {question}
Answer: """
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain

n_gpu_layers = 15  # Change accordint to your GPU VRAM.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Chip.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=model_id,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=False,
    streaming=True
)


# Create a retrieval-based question-answering (QA) chain
# based on a predefined chain type "stuff"
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff",
                                       retriever=retriever,
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                                       verbose=False)


# print('ok1')
# Function to generate output based on a query
def make_output(question):
    answer = qa_chain({"query": question})
    raw_result = answer["result"]
    # print(f"Raw answer: {raw_result}")
    # Split the result at the occurrence of "Unhelpful Answers"
    result = raw_result.split("BBYE!!")[0].strip()
    
    return result


# Function to modify the output by adding spaces between each word with a delay
def modify_output(input):
    # Iterate over each word in the input string
    for text in input.split():
        # Yield the word with an added space
        yield text + " "
        time.sleep(0.001)


        
# Example query
# query = "What is the capital of France?"
# result = make_output(query)
# print(result)