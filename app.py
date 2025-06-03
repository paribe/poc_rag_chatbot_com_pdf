import os
import tempfile
import pickle
import streamlit as st
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np

# ✅ Configuração da página
st.set_page_config(
    page_title="Chat PyGPT",
    page_icon="📄",
    layout="wide"
)

# ✅ Carrega variáveis de ambiente
load_dotenv()

# ✅ Configuração da chave da API do GROQ
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("Por favor, configure a variável GROQ_API_KEY nos secrets do Streamlit")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

# ✅ Configurações fixas
DEFAULT_MODEL = "llama3-8b-8192"
VECTOR_STORE_FILE = "vector_store.pkl"

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
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(documents=docs)
        return chunks
    except Exception as e:
        st.error(f"Erro ao processar PDF: {str(e)}")
        return []

# ✅ Carrega vetor store existente
@st.cache_data
def load_existing_vector_store():
    try:
        if os.path.exists(VECTOR_STORE_FILE):
            with open(VECTOR_STORE_FILE, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.warning(f"Não foi possível carregar base existente: {str(e)}")
    return None

# ✅ Salva vetor store
def save_vector_store(vector_store):
    try:
        with open(VECTOR_STORE_FILE, 'wb') as f:
            pickle.dump(vector_store, f)
    except Exception as e:
        st.warning(f"Não foi possível salvar base: {str(e)}")

# ✅ Cria ou atualiza vetor store
def create_or_update_vector_store(chunks, existing_store=None):
    try:
        embeddings = get_embeddings()
        
        if existing_store and hasattr(existing_store, 'add_documents'):
            # Adiciona novos documentos ao store existente
            existing_store.add_documents(chunks)
            vector_store = existing_store
        else:
            # Cria novo vector store
            if chunks:
                vector_store = FAISS.from_documents(chunks, embeddings)
            else:
                return None
        
        save_vector_store(vector_store)
        return vector_store
        
    except Exception as e:
        st.error(f"Erro ao criar/atualizar vector store: {str(e)}")
        return existing_store

# ✅ Faz a pergunta ao modelo
def ask_question(query, vector_store):
    try:
        llm = ChatGroq(model=DEFAULT_MODEL, temperature=0.1)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        system_prompt = """
        Você é um assistente especializado em responder perguntas baseadas em documentos PDF.
        
        INSTRUÇÕES:
        - Use APENAS o contexto fornecido para responder
        - Se a informação não estiver no contexto, diga claramente que não encontrou
        - Seja preciso e cite partes relevantes do documento quando possível
        - Responda em português brasileiro
        - Use formatação markdown para melhor legibilidade
        
        CONTEXTO DOS DOCUMENTOS:
        {context}
        """

        messages = [("system", system_prompt)]
        
        # Adiciona histórico da conversa
        for message in st.session_state.messages[-10:]:  # Últimas 10 mensagens
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
        return response.get("answer", "Desculpe, não consegui processar sua pergunta.")
        
    except Exception as e:
        return f"Erro ao processar pergunta: {str(e)}"

# ✅ Interface principal
st.header("🤖 Chat com seus documentos PDF")
st.markdown("*Powered by GROQ & HuggingFace*")

# ✅ Inicializa vetor store
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = load_existing_vector_store()

# ✅ Sidebar para upload
with st.sidebar:
    st.header("📄 Upload de Documentos")
    
    uploaded_files = st.file_uploader(
        label="Selecione arquivos PDF",
        type=["pdf"],
        accept_multiple_files=True,
        help="Arraste e solte ou clique para selecionar"
    )

    if uploaded_files:
        with st.spinner("🔄 Processando documentos..."):
            all_chunks = []
            
            # Barra de progresso
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                st.write(f"📖 Processando: {uploaded_file.name}")
                chunks = process_pdf(uploaded_file)
                all_chunks.extend(chunks)
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            if all_chunks:
                st.session_state.vector_store = create_or_update_vector_store(
                    chunks=all_chunks,
                    existing_store=st.session_state.vector_store
                )
                st.success(f"✅ {len(all_chunks)} chunks processados!")
                st.balloons()
            else:
                st.error("❌ Nenhum conteúdo encontrado nos PDFs")
    
    # Info sobre documentos carregados
    if st.session_state.vector_store:
        st.info("📚 Base de conhecimento ativa")
        if st.button("🗑️ Limpar documentos"):
            st.session_state.vector_store = None
            if os.path.exists(VECTOR_STORE_FILE):
                os.remove(VECTOR_STORE_FILE)
            st.rerun()

# ✅ Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input do usuário
if prompt := st.chat_input("Digite sua pergunta sobre os documentos..."):
    
    if not st.session_state.vector_store:
        st.warning("⚠️ Por favor, faça upload de um PDF primeiro!")
    else:
        # Adiciona mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gera resposta
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analisando documentos..."):
                response = ask_question(prompt, st.session_state.vector_store)
                st.markdown(response)
                
        # Adiciona resposta ao histórico
        st.session_state.messages.append({"role": "assistant", "content": response})

# ✅ Footer
st.markdown("---")
st.markdown("🚀 Desenvolvido com Streamlit + LangChain + GROQ")