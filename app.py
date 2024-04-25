import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Read API key from file
file = open('key.txt')
key = file.read().strip()
file.close()

# Initialize embedding model
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=key, model="models/embedding-001")

# Setting a connection with the ChromaDB
db_connection = Chroma(persist_directory="./rag_db", embedding_function=embedding_model)

# Convert CHROMA db_connection to Retriever Object
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

# Define retrieval function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) 

# Define chat prompt template
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a Helpful AI Bot. 
    Your task is to provide assistance based on the context given by the user. 
    Make sure your answers are relevant and helpful."""),
    HumanMessagePromptTemplate.from_template("Answer the question based on the given context.\nContext:\n{context}\nQuestion:\n{question}\nAnswer:")
])

# Initialize chat model
chat_model = ChatGoogleGenerativeAI(google_api_key=key, model="gemini-1.5-pro-latest")

# Initialize output parser
output_parser = StrOutputParser()

# Define RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

# Streamlit UI
st.title("✨ Question Answer System ✨")

# Add custom CSS to change background color
st.markdown(
    """
    <style>
        body {
            background-color: #f0f2f6; /* Change this to the color you desire */
        }
        div.stTextInput>div.stTextInput>div>input {
            font-size: 18px;
            padding: 8px;
        }
        div.stButton>button {
            font-size: 18px;
            padding: 8px 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

question = st.text_input("What is your question?: 🤔")

if st.button("Submit"):
    if question:
        response = rag_chain.invoke(question)
        st.subheader("Answer:")
        st.write(response)
    else:
        st.warning("📑💡Please enter a question.")
