from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA
import streamlit as st

## https://github.com/langchain-ai/langchain/issues/9918#issuecomment-1698734305
## https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed
   

## Get response from Llama 2 Model
def getResponse(question):
    llm = CTransformers(model='/Users/davidayomide/Downloads/Dev/PdfLlama/models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        # alternatively
                        # model='TheBloke/Llama-2-7B-Chat-GGML',
                        # model_file='llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 600,
                                'temperature': 0.01,
                                'context_length':2048})
    # Prompt template
    qa_template = """Use the following pieces of information to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

                {context}
    
                Question: {question}

                Only return the helpful answer below and nothing else.
                Helpful answer:
                """

    prompt = PromptTemplate(input_variables=['context','question'],
                                          template=qa_template)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})

    CHROMA_PATH = "chroma"

    vectordb = Chroma(embedding_function=embeddings,
                      persist_directory=CHROMA_PATH)

    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           retriever=vectordb.as_retriever(
                                               search_kwargs={"k": 2},
                                               search_type='similarity'),
                                    return_source_documents=True,
                                    chain_type="stuff",
                                    chain_type_kwargs={'prompt': prompt})

    response = qa_chain.invoke({'query': question})
    print(response["result"])
    return response["result"]

## UI using Streamlit
st.set_page_config(page_title='PDF Llama',
                   page_icon='🦙',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header('PDF Llama 🦙')

# Input
question = st.text_input("What answers do you seek?")
submit = st.button("Generate🪄")

# Output
if submit:
    result = getResponse(question)
    st.write(result)