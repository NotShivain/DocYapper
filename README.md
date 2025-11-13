# DocYapper: Chat with Your Docs!

**DocYapper** enables you to interactively chat with documents, research papers, or articles using advanced LLM capabilities enhanced with **Retrieval-Augmented Generation (RAG)**.

---

## ⚙️ Setup Instructions

### 1. Generate API Key
This app utilizes **Groq** for LLM inference.  
Generate your own API key from:  
[https://console.groq.com/keys](https://console.groq.com/keys)

---

### 2. Create and Activate a Virtual Environment
It is recommended to use a virtual environment for this project.

```bash
python -m venv venv
venv/Scripts/activate
```
After activating the environment, install all dependencies:
```bash
pip install -r requirements.txt
```
### 3. Configure Secrets
**MANDATORY STEP**
Create a folder named **.streamlit** in the root directory.
Inside this folder, create a file named **secrets.toml**
Add your Groq Api Key in the following format:

**GROQ_API_KEY = "your_api_key"**

Streamlit will automatically fetch the key from this file.

### 4. Run the Project
Setup complete, now to run the project run the following command:
```bash
streamlit run app.py
```
