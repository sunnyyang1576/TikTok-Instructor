__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.embeddings import QianfanEmbeddingsEndpoint
import dotenv
import os
from langchain.retrievers import EnsembleRetriever
from utils import load_dense_retriever,load_sparse_retriever,combine_retrieval_context
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from prompt import rag_prompt_template,simple_prompt_template,rewrite_prompt_template
from langchain_openai import ChatOpenAI


@st.cache_resource
def load_retriever(weight,k):
    dotenv.load_dotenv()
    WX_API_KEY = os.getenv("WX_API_KEY")
    WX_SECRET_KEY = os.getenv("WX_SECRET_KEY")

    embeddings = QianfanEmbeddingsEndpoint(
        qianfan_ak=WX_API_KEY,
        qianfan_sk=WX_SECRET_KEY,
        model="bge-large-en"
    )

    persist_directory = "vectordb/original"
    dense_retriever = load_dense_retriever(persist_directory, embeddings, k=k)
    sparse_retriever = load_sparse_retriever(persist_directory + "/raw/raw_text.csv", k=k)

    ensemble_retriever = EnsembleRetriever(retrievers=[sparse_retriever, dense_retriever],
                                           weights=weight)

    return ensemble_retriever

@st.cache_resource
def load_llm(api_key,model):
    if model == "GPT3.5":
        llm = ChatOpenAI(openai_api_key=api_key,
                   model="gpt-3.5-turbo-0125")
    elif model == "GPT4":
        llm = ChatOpenAI(openai_api_key=api_key,
                   model="gpt-4-0125-preview")
    return llm

@st.cache_resource
def query_retrival(query,_ensemble_retriever):

    docs = _ensemble_retriever.get_relevant_documents(query)

    return docs


def show_document(document,page):
    show=True
    document_name = document.replace("text","") + ".pdf"
    st.session_state.show_doc = show
    st.session_state.document_name = document_name
    st.session_state.page = page

def reset_session_state():
    st.session_state.answer=None
    st.session_state.content = ""
    st.session_state.source = []
    st.session_state.context = ""
    st.session_state.show_doc = False
    st.session_state.document_name = ""
    st.session_state.page = 0


st.set_page_config(page_title="TikTok Instructor", page_icon="📖", layout="wide")
st.title('Your TikTok Operation Instructor.')
st.write("You can ask any questions related to operation of TikTok with reference to the document.")


### Set up model parameters
with st.sidebar:
    gpt_api_key = st.text_input(label="Please Enter Your API Key Here.")
    model = st.selectbox("LLM",("GPT3.5","GPT4"),index=None,placeholder="Select Model to use")
    if gpt_api_key and model:
        llm = load_llm(gpt_api_key,model)

    use_rag = st.toggle('Activate RAG')
    if use_rag:
        k = st.slider(label="Number of Documents Retrieved.",min_value=1,max_value=10,step=1,value=1)
        keyword_search_weight = st.slider(label="Weight for keyword search method.",min_value=0.0,max_value=1.0,step=0.1,value=0.1)
        semantic_search_weight = 1-keyword_search_weight
        retriever =load_retriever([keyword_search_weight,semantic_search_weight],k)
        use_hyde = st.toggle('Use Hyde to Rewrite Question.')

    st.button(label="Reset Session State.",on_click=reset_session_state)

if "answer" not in st.session_state:
    st.session_state.answer=None
    st.session_state.content = ""
    st.session_state.source = []
    st.session_state.context = ""
    st.session_state.show_doc = False
    st.session_state.document_name = ""
    st.session_state.page = 0


query = st.chat_input("Ask Question Here.")

if query:
    if use_rag:

        if use_hyde:
            rewrite_prompt = rewrite_prompt_template.format(query=query)
            search_query = llm.invoke(rewrite_prompt).content
        else:
            search_query = query

        docs = query_retrival(search_query,retriever)
        context = combine_retrieval_context(docs)
        prompt = rag_prompt_template.format(context=context,query=query)
        source = [doc.metadata for doc in docs]
    else:
        prompt = simple_prompt_template.format(query=query)
        source = []
        context = ""

    answer = llm.invoke(prompt)

    st.session_state.answer = answer
    st.session_state.content = answer.content
    st.session_state.source = source
    st.session_state.context = context


col1,col2 = st.columns([0.6, 0.4])

with col1:
    st.subheader("Answer:")
    with st.container(height=500,border=True):
        st.write(st.session_state.content)

    st.subheader("Reference:")
    for element in st.session_state.source:
        with st.expander(element["DOCUMENT"]):
            st.write(element["PARENT"])
            on = st.button(label="Show Document",key=element["IDX"],on_click=show_document,args=(element["DOCUMENT"],element["PAGE"]))

with col2:
    st.subheader("Document")
    if st.session_state.show_doc is True:
        with st.container(height=1000,border=True):
            st.write(st.session_state.document_name)
            pdf_viewer("files/"+st.session_state.document_name,
                   width=700,
                   height=800,
                   rendering="unwrap",
                   pages_to_render=[st.session_state.page])



