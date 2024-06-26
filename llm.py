from langchain_community.llms import CTransformers

# Local CTransformers wrapper for Llama-2-7B-Chat
llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',  # Location of downloaded GGML model
                    model_type='llama',  # Model type Llama
                    config={'max_new_tokens': 256,
                            'temperature': 0.01})
