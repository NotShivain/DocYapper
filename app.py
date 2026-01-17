import streamlit as st
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
import requests
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup

# Initialize session state
if "uploaded_pdf" not in st.session_state:
    st.session_state["uploaded_pdf"] = None

if "web_url" not in st.session_state:
    st.session_state["web_url"] = None

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

if "config" not in st.session_state:
    st.session_state["config"] = {"configurable": {"thread_id": "1"}}

if "documents_processed" not in st.session_state:
    st.session_state["documents_processed"] = False

if "extracted_text" not in st.session_state:
    st.session_state["extracted_text"] = ""

# State schema
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Document processing functions
def extract_pdf_text():
    """Extract text from uploaded PDF files"""
    pdf_docs = st.session_state.get("uploaded_pdf")
    
    if not pdf_docs:
        return ""
    
    text = ""
    try:
        files_to_process = pdf_docs if isinstance(pdf_docs, list) else [pdf_docs]
        
        for pdf_file in files_to_process:
            if pdf_file is None:
                continue
            
            pdf_file.seek(0)
            reader = PdfReader(pdf_file)
            
            if not hasattr(reader, 'pages') or reader.pages is None:
                st.warning(f"Could not read pages from {pdf_file.name}")
                continue
            
            for page in reader.pages:
                if page is not None:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        
        return text.strip()
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return ""

def extract_web_text():
    """Fetch and extract text from web URL"""
    url = st.session_state.get("web_url")
    
    if not url:
        return ""
    
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text.strip()
    except Exception as e:
        st.error(f"Error fetching URL: {str(e)}")
        return ""

# Simple LLM agent node
def agent_node(state: AgentState) -> AgentState:
    """Run LLM with document context"""
    
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            groq_api_key=st.secrets["GROQ_API_KEY"]
        )
        
        context_text = st.session_state.get("extracted_text", "")
        messages = list(state.get("messages", []))
        
        # Add context as system message if we have a document
        if context_text and messages:
            has_context = any(
                isinstance(m, SystemMessage) and "document content" in m.content 
                for m in messages
            )
            
            if not has_context:
                # Truncate context to avoid token limits
                context_snippet = context_text[:4000]
                system_msg = SystemMessage(
                    content=f"""You are a helpful assistant. You have access to the following document content:

---
{context_snippet}
---

Answer the user's questions based on this document. If the answer is not in the document, say so."""
                )
                messages.insert(0, system_msg)
        
        # Get response from LLM
        response = llm.invoke(messages)
        
        return {"messages": [response]}
        
    except Exception as e:
        error_msg = AIMessage(content=f"Error: {str(e)}")
        return {"messages": [error_msg]}

# Build the graph
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    return workflow.compile(checkpointer=InMemorySaver())

# Streamlit UI
st.title("Doc Yapper, chat with your docs!")

# Sidebar for document upload
with st.sidebar:
    st.header("📁 Document Source")
    
    doc_type = st.radio("Choose source:", ["PDF Upload", "Web URL"])
    
    if doc_type == "PDF Upload":
        uploaded_pdf = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        
        if uploaded_pdf:
            if st.session_state.uploaded_pdf != uploaded_pdf:
                st.session_state.uploaded_pdf = uploaded_pdf
                st.session_state.web_url = None
                st.session_state.documents_processed = False
                st.session_state.chat_messages = []
                st.session_state.extracted_text = ""
            
            if not st.session_state.documents_processed:
                with st.spinner("Processing PDF..."):
                    text = extract_pdf_text()
                    if text:
                        st.session_state.extracted_text = text
                        st.session_state.documents_processed = True
                        st.success(f"✅ Extracted {len(text)} characters")
                    else:
                        st.error("Failed to extract text from PDF")
    
    else:  # Web URL
        web_url = st.text_input("Enter web URL")
        
        if web_url and st.button("Load URL"):
            if st.session_state.web_url != web_url:
                st.session_state.web_url = web_url
                st.session_state.uploaded_pdf = None
                st.session_state.documents_processed = False
                st.session_state.chat_messages = []
                st.session_state.extracted_text = ""
            
            if not st.session_state.documents_processed:
                with st.spinner("Fetching web page..."):
                    text = extract_web_text()
                    if text:
                        st.session_state.extracted_text = text
                        st.session_state.documents_processed = True
                        st.success(f"✅ Extracted {len(text)} characters")
                    else:
                        st.error("Failed to fetch web page")
    
    if st.session_state.extracted_text:
        st.info(f"📄 Document loaded ({len(st.session_state.extracted_text)} chars)")
        
        if st.button("Clear Document"):
            st.session_state.uploaded_pdf = None
            st.session_state.web_url = None
            st.session_state.documents_processed = False
            st.session_state.chat_messages = []
            st.session_state.extracted_text = ""
            st.rerun()
        # New Chat button
    st.divider()
    if st.button("🆕 New Chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()
    
    # Watermark at the bottom
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #888; font-size: 0.85em; padding: 10px;'>
            Made with ❤️ by Shivain
        </div>
        """,
        unsafe_allow_html=True
    )

# Initialize graph
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

# Main chat area
st.subheader("💬 Chat")

# Display chat messages
for msg in st.session_state.chat_messages:
    if isinstance(msg, (HumanMessage, SystemMessage)):
        if not (isinstance(msg, SystemMessage) and "document content" in msg.content):
            with st.chat_message("user"):
                st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Chat input
if prompt := st.chat_input("Ask about the document..."):
    if not st.session_state.extracted_text:
        st.warning("⚠️ Please upload a PDF or load a web URL first.")
    else:
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Add to messages
        user_msg = HumanMessage(content=prompt)
        st.session_state.chat_messages.append(user_msg)
        
        # Invoke agent
        with st.spinner("🤔 Thinking..."):
            try:
                result = st.session_state.graph.invoke(
                    {"messages": st.session_state.chat_messages},
                    st.session_state.config
                )
                
                # Update messages
                if result and "messages" in result:
                    st.session_state.chat_messages = list(result["messages"])
                    
                    # Display assistant response
                    last_message = result["messages"][-1]
                    if isinstance(last_message, AIMessage):
                        with st.chat_message("assistant"):
                            st.write(last_message.content)
                            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
        
        st.rerun()