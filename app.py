import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="InsightDoc AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD API KEY ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- CUSTOM CSS FOR PROFESSIONAL UI ---
st.markdown("""
<style>
    .stChatFloatingInputContainer {bottom: 20px;}
    .reportview-container {
        margin-top: -2em;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- FUNCTIONS ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # Use local embeddings (CPU) to save costs and avoid rate limits
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_gemini_response(question, context):
    prompt = f"""
    You are an intelligent document assistant. Your task is to answer the question based strictly on the provided Context.
    
    Guidelines:
    1. If the answer is in the context, provide a clear, concise, and professional answer.
    2. If the answer is NOT in the context, politely state that the document does not contain that information. Do not hallucinate.
    3. Format your answer using bullet points if listing items.

    Context:
    {context}
    
    Question: 
    {question}
    
    Answer:
    """
    model = genai.GenerativeModel("gemini-2.5-flash") # Updated to latest model
    response = model.generate_content(prompt)
    return response.text

# --- MAIN APP LOGIC ---
def main():
    # 1. Sidebar Design
    with st.sidebar:
        st.title("InsightDoc Hub")
        st.markdown("---")
        st.write("Upload your PDFs to start chatting.")
        
        pdf_docs = st.file_uploader("", accept_multiple_files=True, type=['pdf'])
        
        if st.button("🚀 Process Documents", use_container_width=True):
            if pdf_docs:
                with st.spinner("Analyzing documents..."):
                    # Progress bar for visual feedback
                    progress_bar = st.progress(0)
                    
                    raw_text = get_pdf_text(pdf_docs)
                    progress_bar.progress(30)
                    
                    text_chunks = get_text_chunks(raw_text)
                    progress_bar.progress(60)
                    
                    get_vector_store(text_chunks)
                    progress_bar.progress(100)
                    
                    time.sleep(0.5) # Aesthetic pause
                    st.success("Analysis Complete! You can now chat.")
                    st.session_state.docs_processed = True
            else:
                st.warning("Please upload a PDF file first.")
        
        st.markdown("---")
        st.markdown("### System Status")
        if "docs_processed" in st.session_state and st.session_state.docs_processed:
            st.success("🟢 System Ready")
        else:
            st.info("⚪ Waiting for Upload")
            
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # 2. Main Chat Interface
    st.title("InsightDoc AI")
    st.markdown("#### Chat with your documents using Local RAG & Gemini")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Check if docs are processed
        if "docs_processed" not in st.session_state:
            st.error("⚠️ Please upload and process documents from the sidebar first.")
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Thinking..."):
                    try:
                        # Load Vector DB
                        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                        docs = new_db.similarity_search(prompt)
                        context_text = "\n\n".join([doc.page_content for doc in docs])
                        
                        # Get AI Response
                        full_response = get_gemini_response(prompt, context_text)
                        
                        # Simulate typing effect
                        message_placeholder.markdown(full_response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()