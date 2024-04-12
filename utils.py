from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from prompts import qa_template
from llm import llm
import os

# Wrap prompt template in a PromptTemplate object


def set_qa_prompt():
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])
    return prompt


# Build RetrievalQA object
def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(
                                           search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt})
    return dbqa


#     return dbqa
def setup_dbqa():
    # Check current working directory
    print('Current working directory:', os.getcwd())

    # Check if the file exists
    file_path = r'/Users/davidayomide/Downloads/Dev/PdfLlama/vectorstore/faiss_db'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})

    # Load FAISS index
    vectordb = FAISS.load_local(
        file_path, embeddings, allow_dangerous_deserialization=True)

    # Set up retrieval QA
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa
