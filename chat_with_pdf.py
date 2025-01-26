import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain   
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

load_dotenv()


if __name__ == "__main__":
        print("hi")

        pdf_path = "C:/Users/satvi/OneDrive/Desktop/langchain_medium/Langchain_Medium/react.pdf"
        loader = PyPDFLoader(file_path=pdf_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        docs = text_splitter.split_documents(documents=documents)

        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        # storing and loading
        vectorstore.save_local("faiss_index_react")

        new_vectorstore = FAISS.load_local(
                "faiss_index_react", embeddings, allow_dangerous_deserialization=True
        )
        print("embeddings loaded")

        # RAG implementation
        chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

        combine_docs_chain = create_stuff_documents_chain(OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY")), chat_prompt)

        retrieval_chain = create_retrieval_chain(
        retriever=new_vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
        )

        res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
        print(res["answer"])

