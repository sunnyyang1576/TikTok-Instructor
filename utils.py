import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever


def chunk_one_file(file,path,chunker):
    document = file.replace(".csv","")
    text_df = pd.read_csv(path + file, index_col="page")

    all_text_list = []
    all_metadata_list = []
    for page in text_df.index:
        text = text_df.loc[page, "content"]
        chunker.text = text
        meta_data_list, child_list_flatten = chunker.chunk(seperator="####New Title:\n\n",minimum_length=100)
        all_text_list += child_list_flatten

        for metadata in meta_data_list:
            metadata["PAGE"] = page
            metadata["DOCUMENT"] = document
            all_metadata_list.append(metadata)

    return all_metadata_list, all_text_list



def load_dense_retriever(vector_db_path,embeddings,k):

    vectordb = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
    dense_retriever = vectordb.as_retriever(search_kwargs={"k": k})

    return dense_retriever



def load_sparse_retriever(raw_text_path,k):
    raw_text = pd.read_csv(raw_text_path)
    meta_data_list = raw_text.to_dict(orient="records")
    text_list = [dic.pop("TEXT") for dic in meta_data_list]
    sparse_retriever = BM25Retriever.from_texts(text_list, metadatas=meta_data_list, k=k)

    return sparse_retriever



def combine_retrieval_context(docs):

    text_list = [doc.metadata["PARENT"] for doc in docs]
    context = "\n\n——————\n\n".join(text_list)
    return context