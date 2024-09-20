import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()




if __name__ == "__main__":
    
    #Loading the document
    print("Ingesting...")

    loader = TextLoader("C:/Users/satvi/OneDrive/Desktop/langchain_medium/ingestion.py")
    document = loader.load()

    #splitting the document
    print("Splitting...")
    
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    print(f"created {len(texts)} chunks")


    #create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))


    #storing in pinecone
    print("ingesting embeddings in pinecone")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME"))