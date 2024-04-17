from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import streamlit as st

# https://github.com/langchain-ai/langchain/issues/9918#issuecomment-1698734305
   

## Get response from Llama 2 Model
def getResponse(question):
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        # alternatively
                        # model='TheBloke/Llama-2-7B-Chat-GGML',
                        # model_file='llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})
    # Prompt template
    qa_template = """Use the following pieces of information to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """

    prompt = PromptTemplate(template=qa_template,
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
                                    chain_type_kwargs={'prompt': prompt})

    response = qa_chain({'question': question})
    print(response["result"])
    return response["result"]

## UI using Streamlit
st.set_page_config(page_title='PDF Llama',
                   page_icon='🦙',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header('PDF Llama 🦙')

# Input
# context = st.file_uploader('Pick a pdf')
question = st.text_input("What answers do you seek?")
submit = st.button("Generate🪄")

# Output
if submit:
    result = getResponse(question)
    st.write(result)