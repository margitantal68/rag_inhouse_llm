import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings  

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def setup_qa_system(file_path):
    # Load and split the PDF document
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()
    print(f"The original document has been split into {len(docs)} documents")

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    print(f"The original document has been split into {len(chunks)} chunks")
    
    embeddings = OllamaEmbeddings(base_url="http://192.168.11.102:11500", model="nomic-embed-text")
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Create a retriever from the vector store
    retriever = vector_store.as_retriever()

    llm = OllamaLLM(base_url="http://192.168.11.102:11500", model="mistral-nemo:12b-instruct-2407-q8_0") 
    # llm = OllamaLLM(base_url="http://192.168.11.102:11500", model="qwen2.5:7b")

    # Define the system prompt
    system_prompt = (
        "Use only the given contexts to answer the question. "
        "If the question cannot be answered by the given contexts, say I don't know. "
        "Use three sentences maximum and keep the answer concise. "
        "Context: {context}"
        "The question should be answered using the same language it was asked in."
    )
    prompt = ChatPromptTemplate.from_messages(
        [ ("system", system_prompt), ("human", "{input}")]
    )

    # Create the document chain and retrieval chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    return chain


if __name__ == '__main__':
    EN_DOCUMENT = "data/RAG_Evaluation_Survey.pdf"
    # HU_DOCUMENT = "data/hatarozatok_249.pdf"


    DOCUMENT = EN_DOCUMENT
    print(f"Using document: {DOCUMENT}")
    qa_chain = setup_qa_system(DOCUMENT)

    while True:
        question = input("\n\n####Ask a question (or type exit): ")
        if question.lower() == 'exit':
            break
    
        answer = qa_chain.invoke({"input": question})
        # Print the context in a structured format
        print("\n\n####Retrieved Context:")
        for doc in answer.get('context', []):
            print(f"\tDocument source: {doc.metadata.get('source', 'No source found')}")
            print(f"\tDocument page: {doc.metadata.get('page', 'No page found')}")
            print(f"\tDocument text: {doc.page_content}")
            print("-" * 80)

        print('\n\n####Answer: ')
        print(answer.get('answer', 'No answer found'))
