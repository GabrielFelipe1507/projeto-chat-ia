import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory 
import os
from dotenv import load_dotenv
import time 

# Importa as funções do db
from db import (
    listar_conversas, 
    criar_nova_conversa, 
    carregar_mensagens, 
    salvar_mensagem,
    deletar_conversa,
    atualizar_titulo_conversa
)

# Carrega .env
load_dotenv()

# --- Configuração do LLM ---
try:
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-preview-09-2025", 
                                 google_api_key=os.getenv("GEMINI_API_KEY"),
                                 convert_system_message_to_human=True)
except Exception as e:
    st.error(f"Erro LLM: {e}")
    st.stop()

# --- Chain de Título ---
try:
    prompt_titulo_template = ChatPromptTemplate.from_template(
        "Gere um título muito curto e descritivo (máximo 5 palavras, idealmente 2-3) para uma conversa de chatbot que começa com a seguinte mensagem do usuário: '{primeira_mensagem}'. O título deve resumir o tópico principal. Responda APENAS com o título, sem introduções como 'Título:', sem aspas e sem pontuação final."
    )
    chain_gerar_titulo = RunnablePassthrough.assign(primeira_mensagem=lambda x: x['input']) | prompt_titulo_template | llm 
    print("DEBUG: Chain de título criada.")
except Exception as e:
    st.warning(f"Aviso: Chain de título não criada: {e}")
    chain_gerar_titulo = None

# --- Configuração da Memória e Chain Principal ---
def get_session_history(session_id):
    if session_id is None: 
        return ChatMessageHistory() 
    mensagens_do_banco = carregar_mensagens(session_id) 
    history = ChatMessageHistory()
    for msg in mensagens_do_banco:
        history.add_message(msg)
    return history

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente prestativo. Responda às perguntas do usuário da forma mais completa e educada possível."),
        MessagesPlaceholder(variable_name="history"), 
        ("human", "{input}"), 
    ]
)

try:
    chain_with_memory = RunnableWithMessageHistory(
        prompt_template | llm, 
        get_session_history,    
        input_messages_key="input", 
        history_messages_key="history", 
    )
except Exception as e:
    st.error(f"Erro Chain Memória: {e}")
    st.stop()

# --- Interface Streamlit ---
st.set_page_config(page_title="Chatbot com MySQL v4", layout="wide")
st.title("Meu Chatbot com Gemini e MySQL v4 💾🤖")

# --- Sidebar ---
st.sidebar.title("Minhas Conversas")

if st.sidebar.button("➕ Novo Chat", key="novo_chat_sidebar_button"):
    st.session_state.conversa_ativa_id = None
    st.rerun() 

try:
    lista_de_conversas = listar_conversas() 
except Exception as e:
    st.sidebar.error(f"Erro ao listar conversas: {e}")
    lista_de_conversas = []

if "conversa_ativa_id" not in st.session_state:
    st.session_state.conversa_ativa_id = None 

st.sidebar.divider() 
st.sidebar.markdown("**Histórico:**")
if not lista_de_conversas:
    st.sidebar.info("Nenhuma conversa ainda.")
else:
    conversations_container = st.sidebar.container(height=300)
    with conversations_container:
        for conversa in lista_de_conversas: 
            conversa_id = conversa['id']
            titulo_display = conversa.get('titulo') or f'Conversa ID {conversa_id}' 
            col1, col2 = st.columns([0.85, 0.15], gap="small") 
            with col1:
                if st.button(titulo_display, key=f"conversa_{conversa_id}", use_container_width=True):
                    st.session_state.conversa_ativa_id = conversa_id
                    st.rerun() 
            with col2:
                 if st.button("🗑️", key=f"delete_{conversa_id}", help=f"Deletar conversa {conversa_id}", use_container_width=True):
                     try:
                        if deletar_conversa(conversa_id):
                            st.toast(f"Conversa {conversa_id} deletada.", icon="✅") 
                            if st.session_state.get("conversa_ativa_id") == conversa_id:
                                st.session_state.conversa_ativa_id = None
                            time.sleep(0.5) 
                            st.rerun() 
                        else:
                            st.error(f"Erro ao deletar conversa {conversa_id}.")
                     except Exception as e:
                         st.error(f"Erro inesperado ao deletar: {e}")

# --- Área Principal ---
active_chat_id = st.session_state.get("conversa_ativa_id") 

# Exibe o histórico (SE houver chat ativo)
if active_chat_id:
    try:
        chat_history_para_exibir = get_session_history(active_chat_id)
        for message in chat_history_para_exibir.messages:
            role = "ai" if isinstance(message, AIMessage) else "human"
            with st.chat_message(role):
                st.markdown(message.content)
    except Exception as e:
        st.error(f"Erro ao carregar histórico para exibição: {e}")
else:
    st.info("⬅️ Selecione uma conversa na barra lateral ou digite abaixo para iniciar um novo chat.")

# --- INPUT ÚNICO ---
# Define o placeholder dinamicamente
placeholder = "Digite sua mensagem..." if active_chat_id else "Digite sua primeira mensagem para iniciar um novo chat..."

if prompt := st.chat_input(placeholder, key="chat_input_principal"):
    
    # Lógica se JÁ EXISTE um chat ativo
    if active_chat_id:
        with st.chat_message("human"):
            st.markdown(prompt)
        if not salvar_mensagem(active_chat_id, "human", prompt):
            st.error("Erro ao salvar sua mensagem.")
            st.stop()
        try:
            with st.spinner("Digitando..."):
                response = chain_with_memory.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": active_chat_id}}
                )
            if response and hasattr(response, 'content') and response.content.strip():
                if not salvar_mensagem(active_chat_id, "ai", response.content):
                    st.error("Erro ao salvar a resposta da IA.")
                st.rerun() # Recarrega para mostrar a resposta salva
            else:
                 st.warning("O LLM retornou uma resposta vazia.")
        except Exception as e:
            st.error(f"Erro ao processar mensagem: {e}")

    # Lógica se é um NOVO chat
    else: 
        try:
            novo_id = criar_nova_conversa()
        except Exception as e:
            st.error(f"Erro ao criar nova conversa: {e}")
            novo_id = None
            
        if novo_id:
            st.session_state.conversa_ativa_id = novo_id # Define como ativa ANTES de salvar/invocar
            print(f"DEBUG: Novo chat criado (ID:{novo_id}). Mensagem: {prompt}")
            if not salvar_mensagem(novo_id, "human", prompt):
                st.error("Erro ao salvar sua primeira mensagem.")
                st.stop()
            try:
                with st.spinner("Digitando..."):
                    response = chain_with_memory.invoke(
                        {"input": prompt},
                        config={"configurable": {"session_id": novo_id}}
                    )
                if response and hasattr(response, 'content') and response.content.strip():
                    if not salvar_mensagem(novo_id, "ai", response.content):
                        st.error("Erro ao salvar a primeira resposta da IA.")
                    
                    # Tenta gerar e salvar título
                    if chain_gerar_titulo:
                        try:
                            with st.spinner("Gerando título..."):
                                titulo_response = chain_gerar_titulo.invoke({"input": prompt})
                            if titulo_response and hasattr(titulo_response, 'content') and titulo_response.content.strip():
                                novo_titulo = titulo_response.content
                                if not atualizar_titulo_conversa(novo_id, novo_titulo):
                                    st.warning("Não foi possível salvar o título gerado.")
                            else:
                                st.warning("LLM não gerou um título válido.")
                        except Exception as e_titulo:
                            st.warning(f"Erro ao gerar título: {e_titulo}")
                    
                    print("DEBUG: Recarregando após novo chat.")
                    time.sleep(0.5)
                    st.rerun() # Recarrega para mostrar tudo
                else:
                    st.warning("LLM retornou resposta vazia na primeira mensagem.")
            except Exception as e:
                st.error(f"Erro ao processar primeira mensagem: {e}")
        else:
            st.error("Falha ao criar nova conversa no banco.")

