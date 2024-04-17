"""
This script creates a database of information gathered from local text files.
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA


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

raw_pdf = PyPDFLoader(
    '/Users/davidayomide/Downloads/Dev/PdfLlama/manu-20f-2022-09-24.pdf')
pages = raw_pdf.load()


# text_splitter = CharacterTextSplitter(chunk_size=800,
#                                           separator='\n\n',
#                                            chunk_overlap=50,
#                                            length_function= len)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, 
                                               chunk_overlap=0,
                                               separators=["\n\n", "\n", " ", ""])
all_splits = text_splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})

# create and save the local database
db = FAISS.from_documents(documents=all_splits, embedding=embeddings)
db.save_local("vectorstore/faiss_db")

question = "What is the title of the file?"
# docs = db.similarity_search(question, k=3)
# print(docs)

llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                    # alternatively
                    # model='TheBloke/Llama-2-7B-Chat-GGML',
                    # model_file='llama-2-7b-chat.ggmlv3.q8_0.bin',
                    model_type='llama',
                    config={'max_new_tokens': 256,
                            'temperature': 0.01})
qa_template = """Use the following pieces of information to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """

qa_prompt = PromptTemplate(template=qa_template,
                        input_variables=['question'])

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': 'cpu'})

vectordb = FAISS.load_local('/Users/davidayomide/Downloads/Dev/PdfLlama/vectorstore/faiss_db',
                            embeddings=embeddings, allow_dangerous_deserialization=True)

qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       retriever=vectordb.as_retriever(
                                           search_kwargs={"k": 3}),
                                       return_source_documents=True,
                                       chain_type="stuff",
                                       chain_type_kwargs={'prompt': qa_prompt})

response = qa_chain({'question': question})
print(response["result"])
