from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()




loader = PyPDFLoader("research-papers/A Comprehensive Survey of Retrieval-Augmented Generation (RAG)-- Evolution, Current Landscape and Future Directions.pdf")
pages = loader.load_and_split()


# Save all page text into a single .txt file
with open("pdf_pages_output.txt", "w", encoding="utf-8") as f:
    for i, page in enumerate(pages):
        f.write(f"\n\n=== Page {i+1} ===\n\n")
        f.write(page.page_content)



text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)


retriever = vectorstore.as_retriever()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


llm = ChatOpenAI()


template = """
SYSTEM: You are a question-answering assistant.
Use only the provided context to answer.
If you don’t know, respond with “I don’t know.”
QUESTION: {question}
CONTEXT:
{context}
"""
prompt = PromptTemplate.from_template(template)



chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


print(chain.invoke("What is BM25?"))


print("hi")