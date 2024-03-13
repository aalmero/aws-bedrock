import os
from dotenv import load_dotenv

#embedding
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import BedrockEmbeddings

#load environment variables
load_dotenv()

loader = PyPDFLoader(file_path="2022-Shareholder-Letter.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter( #create a text splitter
        separators=["\n\n", "\n", ".", " "], #split chunks at (1) paragraph, (2) line, (3) sentence, or (4) word, in that order
        chunk_size=1000, #divide into 1000-character chunks using the separators above
        chunk_overlap=100 #number of characters that can overlap with previous chunk
 )

docs = text_splitter.split_documents(documents)

embeddings = BedrockEmbeddings(
    credentials_profile_name="sre-sandbox-genai", region_name="us-east-1"
)

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
    host=os.environ.get("PGVECTOR_HOST", "localhost"),
    port=int(os.environ.get("PGVECTOR_PORT", "5432")),
    database=os.environ.get("PGVECTOR_DATABASE", "postgres"),
    user=os.environ.get("PGVECTOR_USER", "postgres"),
    password=os.environ.get("PGVECTOR_PASSWORD", "postgres"),
)

COLLECTION_NAME = "share_holder_letter"

db = PGVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

#using existing vector store
""" 
store = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)
 """
query = "What is the company's generative AI strategy?"

docs_with_score = db.similarity_search_with_score(query)


for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)

