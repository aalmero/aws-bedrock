import argparse
import os
from dotenv import load_dotenv

from langchain_community.vectorstores.pgvector import PGVector
from langchain.llms.bedrock import Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import BedrockChat
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain

from langchain.chains import ConversationalRetrievalChain



PROMPT_TEMPLATE_1 = """
Answer the question base on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

PROMPT_TEMPLATE = """
Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Assistant:
"""


# get data from vector store
def get_data(collection: str):

    #create embeddings
    embeddings = BedrockEmbeddings(
        credentials_profile_name="sre-sandbox-genai", 
        region_name="us-east-1")
    
    CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
        host=os.environ.get("PGVECTOR_HOST", "localhost"),
        port=int(os.environ.get("PGVECTOR_PORT", "5432")),
        database=os.environ.get("PGVECTOR_DATABASE", "postgres"),
        user=os.environ.get("PGVECTOR_USER", "postgres"),
        password=os.environ.get("PGVECTOR_PASSWORD", "postgres"),
    )
    
    #COLLECTION_NAME = collection

    data = PGVector(
        collection_name=collection,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )

    return data


def get_model():
    #create llm
    llm = Bedrock(
        credentials_profile_name="sre-sandbox-genai", 
        region_name="us-east-1", 
        model_id="anthropic.claude-v2",
        model_kwargs={"max_tokens_to_sample": 200})
    
    return llm


def main():
    load_dotenv()

    #create simple cli
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Query Text.")
    args = parser.parse_args()
    query_text = args.query_text

    #get data using collection
    #db = get_data("tax_info")
    db = get_data("rb2")

    #search db
    #results = db.similarity_search_with_relevance_scores(query_text, k=3)

    #print(f"results: {results}")

    llm = get_model()

    #generative question-answering 
    # returns {"question": str, "answer": str}
    #qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever = db.as_retriever())
    #response = qa.invoke(query_text)
    #print(f"response: {response}")

    #using LCEL
    #create a chain that takes a question and the retrieved documents and generate answer
    #document_chain = create_stuff_documents_chain(llm, prompt_template)
    #passing document directly
    #document_chain.invoke({"input": query_text, "context": [Document(page_content="langsmith can let you visualize test results")]})
    #or use a retriever
    #retrieval_chain = create_retrieval_chain(db.as_retriever(), document_chain)
    #response = retrieval_chain.invoke({"input": query_text})
    #print(response["answer"])

    #generative question-answering with sources, using citations
    # returns {"question": str, "answer": str, "sources": str}
    #qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever = db.as_retriever())
    #response = qa.invoke(query_text)
    #print(f"response: {response}")

    #using RetrievalQA with customizable option
    # returns {"query": str, "result": str, "source_documents": [Documents]}
    # Anthropic Claude as the LLM under Amazon Bedrock, 
    # this particular model performs best if the inputs are provided under Human: 
    # and the model is requested to generate an output after Assistant:
    prompt_template = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
   
    """
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )

    result = qa.invoke(query_text) 
    print(f"result: {result}")
     
    response = result["result"] # answer???
    print(f"response: {response}")

    source = result["source_documents"]
    print(f"source: {source}") 

    #answer = result.get("answer")
    #print(f"answer: {answer}")
    """
    chain = ConversationalRetrievalChain(
                llm=llm,
                retriever=db.as_retriever(),
                return_source_documents=True,
    )


"""
    if(len(results) == 0 or results[0][1] < 0.7):
        print(f"Unable to find matching results.")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, score in results])    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = BedrockChat(credentials_profile_name="sre-sandbox-genai", model_id="anthropic.claude-v2", model_kwargs={"temperature": 0.1})
    #response_text = model.predict(prompt)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
"""
    


if __name__ == "__main__":
    main()