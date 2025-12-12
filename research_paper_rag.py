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



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],  # try to respect paragraphs, then lines, then words
)

chunks = text_splitter.split_documents(pages)


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)


retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,           # how many to return to the LLM
        "fetch_k": 20,    # how many to consider before pruning similar ones
        "lambda_mult": 0.5,  # 0 = diverse, 1 = similar; 0.5 is balanced
    },
)


# ---- Helpers for formatting context & sources ----
def format_docs(docs):
    """Format docs for the LLM, with source labels and pages."""
    formatted = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", None)
        # PyPDFLoader pages are 0-indexed; make them human-friendly
        page_display = page + 1 if isinstance(page, int) else page
        header = f"[Source {i} | Page {page_display}]"
        formatted.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(formatted)

def print_sources(docs):
    """Print concise source citations to the console."""
    print("Sources:")
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", None)
        page_display = page + 1 if isinstance(page, int) else page
        snippet = doc.page_content[:200].replace("\n", " ")
        print(f"[{i}] Page {page_display}: {snippet}...")
    print()  # blank line


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


#QA chain (no retrieval inside; we pass context explicitly)
qa_chain = prompt | llm | StrOutputParser()

#Interactive loop with citations
if __name__ == "__main__":
    print("📚 RAG PDF Chatbot (with sources)")
    print("Ask me anything about the PDF.")
    print("Type 'exit', 'quit', or press Ctrl+C to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
            if question.lower() in {"exit", "quit", "q"}:
                print("Bot: Goodbye! 👋")
                break
            if not question:
                continue

            # Retrieve relevant chunks
            docs = retriever.invoke(question)

            # Build context string for the LLM
            context = format_docs(docs)

            # Get answer
            answer = qa_chain.invoke({"question": question, "context": context})

            # Print answer + sources
            print(f"\nBot: {answer}\n")
            print_sources(docs)

        except KeyboardInterrupt:
            print("\nBot: Goodbye! 👋")
            break




    # Useful Link: https://chatgpt.com/share/693c0cfa-d2f8-8004-9de8-3e0ee3e06406