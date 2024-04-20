"""
This script creates a database of information gathered from local text files.
"""

import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# define what documents to load
# loader = DirectoryLoader("./", glob="*.pdf", loader_cls=PyPDFLoader)

# interpret information in the documents
# documents = loader.load()
# splitter = RecursiveCharacterTextSplitter(chunk_size=500,
#                                           chunk_overlap=50)
# texts = splitter.split_documents(documents)


## Read file and turn to chunks
# pdfReader = PdfReader(
#     '/Users/davidayomide/Downloads/Dev/PdfLlama/manu-20f-2022-09-24.pdf')


# for i, pages in enumerate(pdfReader.pages):
#     raw_text = ''
#     content = pages.extract_text()
#     if content:
#         raw_text += content

DATA_PATH = 'data/books'

raw_data = DirectoryLoader(DATA_PATH, glob="*.md")
pages = raw_data.load()


# text_splitter = CharacterTextSplitter(chunk_size=800,
#                                           separator='\n\n',
#                                            chunk_overlap=50,
#                                            length_function= len)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                               chunk_overlap=0,
                                               add_start_index=True,
                                               length_function=len)
all_splits = text_splitter.split_documents(pages)

print(f"Split {len(pages)} documents into {len(all_splits)} chunks.")

# print(all_splits[10].page_content)
# print(all_splits[10].metadata)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})
print(">>>Embeddings setup completed successfully<<<")

# create and save the local database
# db = FAISS.from_documents(documents=all_splits, embedding=embeddings)
# db.save_local("vectorstore/faiss_db")

CHROMA_PATH = "chroma"

if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

db = Chroma.from_documents(documents=all_splits, 
                           embedding=embeddings, 
                           persist_directory=CHROMA_PATH)
db.persist()

print(f"Save {len(all_splits)} chunks to {CHROMA_PATH}")


## Database test

question = "How did Alice meet the Bunny?"
docs = db.similarity_search_with_relevance_scores(question, k=1)
# # print(all_splits)
print("==========================================================================")
# if len(docs) == 0 or docs[0][1] < 0.7:
#     print("Unable to find matching results")
# else:
print("Some result using similarity search")
print(docs)
print('-------------')

llm = CTransformers(
                    model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
#                     # alternatively
#                     model='TheBloke/Llama-2-7B-Chat-GGML',
#                     model_file='llama-2-7b-chat.ggmlv3.q8_0.bin',
                    model_type='llama',
                    config={'max_new_tokens': 600,
                            'temperature': 0.01,
                            'context_length':2048})
qa_template = """Use the following pieces of information to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Context: {context}

            Question: {question}

            Only return the helpful answer below and nothing else.
            Helpful answer:
            """

qa_prompt = PromptTemplate(input_variables=['context', 'question'],
                           template=qa_template)

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
#                                    model_kwargs={'device': 'cpu'})

# vectordb = FAISS.load_local('/Users/davidayomide/Downloads/Dev/PdfLlama/vectorstore/faiss_db',
#                             embeddings=embeddings, allow_dangerous_deserialization=True)

vectordb = Chroma(CHROMA_PATH, embedding_function=embeddings)

qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       retriever=vectordb.as_retriever(
                                           search_type="similarity",
                                           search_kwargs={"k": 2}),
                                       return_source_documents=True,
                                       chain_type="stuff",
                                       chain_type_kwargs={'prompt': qa_prompt})

response = qa_chain.invoke({'query': question})
print('-------------##########------------')
print("Here is your answer")
print(response['result'])
