import streamlit as st
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain

chat_css = """
<style>
.chat-container {
    max-height: 70vh;
    overflow-y: auto;
    padding: 10px;
    background-color: #f7f7f8;
    border-radius: 10px;
    margin-bottom: 10px;
}
.user-msg {
    background-color: #0b93f6;
    color: white;
    padding: 8px 12px;
    border-radius: 15px 15px 0 15px;
    margin: 5px 0;
    text-align: right;
    width: fit-content;
    float: right;
    clear: both;
}
.bot-msg {
    background-color: #e5e5ea;
    color: black;
    padding: 8px 12px;
    border-radius: 15px 15px 15px 0;
    margin: 5px 0;
    text-align: left;
    width: fit-content;
    float: left;
    clear: both;
}
</style>
"""

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def extract_text_from_link(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        return "\n".join(paragraphs).strip()
    except Exception as e:
        st.error(f"Error fetching article: {e}")
        return ""


def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)


def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embedding=embeddings)


def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=1,
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory)


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("⚠️ Please upload and process a source first!!")
        return

    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for i, msg in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.markdown(f'<div class="user-msg">{msg.content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-msg">{msg.content}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <script>
        const chatContainers = window.parent.document.querySelectorAll('.chat-container');
        if(chatContainers.length > 0){
            chatContainers[chatContainers.length-1].scrollTop = chatContainers[chatContainers.length-1].scrollHeight;
        }
        </script>
        """,
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(page_title="Chat with your docs", page_icon="🤖")
    st.markdown(chat_css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("DocYapper, Your Document/Article Assistant")

    with st.sidebar:
        st.subheader("📄 Source Options")


        option = st.radio("Choose input type:", ("Upload PDFs", "Add Article Link"))

        if option == "Upload PDFs":
            pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
            if st.button("Process PDFs"):
                if not pdf_docs:
                    st.warning("Please upload atleast 1 pdf")
                else:
                    with st.spinner("Processing PDFs... ⏳"):
                        raw_text = get_pdf_text(pdf_docs)
                        chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(chunks)
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("✅ PDFs processed successfully!")

        elif option == "Add Article Link":
            article_url = st.text_input("Enter the article URL:")
            if st.button("Process Article"):
                if not article_url:
                    st.warning("Please provide a correct link")
                else:
                    with st.spinner("Fetching and processing article... 🌐"):
                        article_text = extract_text_from_link(article_url)
                        if article_text:
                            chunks = get_text_chunks(article_text)
                            vectorstore = get_vectorstore(chunks)
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            st.success("✅ Article processed successfully!")

        st.markdown(
            """
            <div style="position: fixed; bottom: 10px; left: 20px; font-size: 14px; color: #999; opacity: 0.7;">
                Made with ❤️ by <b>Shivain</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    user_question = st.text_input("Type your question here and press Enter:", key="chat_input")
    if user_question:
        handle_userinput(user_question)


if __name__ == "__main__":
    main()
