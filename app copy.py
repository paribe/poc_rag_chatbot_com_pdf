import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# ✅ Configuração da página
st.set_page_config(
    page_title="Chat PyGPT",
    page_icon="📄",
)

# ✅ Carrega variáveis de ambiente
load_dotenv()

# ✅ Configuração da chave da API do GROQ
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("Por favor, configure a variável GROQ_API_KEY no arquivo .env ou nos secrets do Streamlit")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

# ✅ Configurações fixas
persist_directory = "db"
DEFAULT_MODEL = "llama3-8b-8192"

# ✅ Embeddings gratuitos do HuggingFace
@st.cache_resource
def get_embeddings():
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    except Exception as e:
        st.error(f"Erro ao carregar embeddings: {str(e)}")
        st.stop()

# ✅ Processamento do PDF
def process_pdf(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        os.remove(temp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=500,
        )
        chunks = text_splitter.split_documents(documents=docs)
        return chunks
    except Exception as e:
        st.error(f"Erro ao processar PDF: {str(e)}")
        return []

# ✅ Carrega vetor store existente
def load_existing_vector_store():
    try:
        if os.path.exists(persist_directory):
            embeddings = get_embeddings()
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
            )
    except Exception as e:
        st.warning(f"Não foi possível carregar base existente: {str(e)}")
    return None

# ✅ Adiciona chunks ao vetor store
def add_to_vector_store(chunks, vector_store=None):
    try:
        embeddings = get_embeddings()
        if vector_store:
            vector_store.add_documents(chunks)
        else:
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_directory,
            )
        return vector_store
    except Exception as e:
        st.error(f"Erro ao adicionar documentos: {str(e)}")
        return vector_store

# ✅ Faz a pergunta ao modelo
def ask_question(query, vector_store):
    try:
        llm = ChatGroq(model=DEFAULT_MODEL)
        retriever = vector_store.as_retriever()

        system_prompt = """
        Use o contexto para responder as perguntas.
        Se não encontrar uma resposta no contexto,
        explique que não há informações disponíveis.
        Responda em formato de markdown e com visualizações
        elaboradas e interativas.
        Contexto: {context}
        """

        messages = [("system", system_prompt)]
        for message in st.session_state.messages:
            messages.append((message.get("role"), message.get("content")))
        messages.append(("human", "{input}"))

        prompt = ChatPromptTemplate.from_messages(messages)

        question_answer_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt,
        )
        chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=question_answer_chain,
        )
        response = chain.invoke({"input": query})
        return response.get("answer")
    except Exception as e:
        return f"Erro ao processar pergunta: {str(e)}"

# ✅ Inicializa vetor store
vector_store = load_existing_vector_store()

st.header("🤖 Chat com seus documentos em PDF")

with st.sidebar:
    st.header("Upload de arquivos 📄")
    uploaded_files = st.file_uploader(
        label="Faça o upload de arquivos PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        with st.spinner("Processando documentos..."):
            all_chunks = []
            for uploaded_file in uploaded_files:
                chunks = process_pdf(file=uploaded_file)
                all_chunks.extend(chunks)
            
            if all_chunks:
                vector_store = add_to_vector_store(
                    chunks=all_chunks,
                    vector_store=vector_store,
                )
                st.sidebar.success("Documentos processados com sucesso!")
            else:
                st.sidebar.error("Erro ao processar documentos.")

# ✅ Controle de sessão
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ✅ Entrada do usuário
question = st.chat_input("Como posso ajudar?")

if vector_store and question:
    for message in st.session_state.messages:
        st.chat_message(message.get("role")).write(message.get("content"))

    st.chat_message("user").write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Buscando resposta..."):
        response = ask_question(
            query=question,
            vector_store=vector_store,
        )

        st.chat_message("ai").write(response)
        st.session_state.messages.append({"role": "ai", "content": response})

elif question and not vector_store:
    st.warning("Por favor, faça upload de um PDF primeiro para começar a conversar!")