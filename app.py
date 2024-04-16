from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
import streamlit as st

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
    qa_template = """You are a therapist that give advice like Yoda from the Star
    wars franchise as such,
    Question: {question}
    Helpful advice:
    """
    # """Use the following pieces of information to answer the user's question.
    # If you don't know the answer, just say that you don't know, don't try to make up an answer.
    # Context: {context}
    # Question: {question}
    # Only return the helpful answer below and nothing else.
    # Helpful answer:
    # """

    prompt = PromptTemplate(template=qa_template, 
                            input_variables=['question'])

    response = llm(prompt.format(question=question))
    print(response)
    return response


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
    st.write(getResponse(question))