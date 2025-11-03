import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
import os
from dotenv import load_dotenv
import time  # Para adicionar um pequeno delay e melhorar a percepção

# Importa as funções do db
from db import (
    listar_conversas,
    criar_nova_conversa,
    carregar_mensagens,
    salvar_mensagem,
    deletar_conversa,
    atualizar_titulo_conversa
)

# Carrega as variáveis de ambiente (GEMINI_API_KEY, DB_HOST, etc.) do arquivo .env

load_dotenv()

# --- Configuração do LLM ---
try:
    # Inicializa o modelo de chat do Google (Gemini)
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-preview-09-2025",
                                 google_api_key=os.getenv("GEMINI_API_KEY"),
                                 convert_system_message_to_human=True)
except Exception as e:
    st.error(f"Erro LLM: {e}")
    st.stop()

# --- 1.1 Mini-Chain para gerar títulos ---
# Esta é uma "chain" (sequência de passos) separada, só para criar títulos
try:
    # 1. Define o template do prompt: o que vamos pedir ao LLM
    prompt_titulo_template = ChatPromptTemplate.from_template(
        "Gere um título muito curto e descritivo (máximo 5 palavras, idealmente 2-3) para uma conversa de chatbot que começa com a seguinte mensagem do usuário: '{primeira_mensagem}'. O título deve resumir o tópico principal. Responda APENAS com o título, sem introduções como 'Título:', sem aspas e sem pontuação final."
    )
    # 2. Define a chain:
    # RunnablePassthrough.assign(...) pega o input original (que será `{'input': '...'}
    # e cria uma nova chave 'primeira_mensagem' com o mesmo valor.
    # O operador | (pipe) "conecta" os passos:
    # 1. Pega o input -> 2. Passa para o prompt_titulo_template -> 3. Passa para o LLM
    chain_gerar_titulo = RunnablePassthrough.assign(
        primeira_mensagem=lambda x: x['input']) | prompt_titulo_template | llm
    print("DEBUG: Chain de título criada.")
except Exception as e:
    st.warning(f"Aviso: Chain de título não criada: {e}")
    chain_gerar_titulo = None

# --- 2. CONFIGURAÇÃO DO BACKEND (Memória e Chain Principal) ---

# Função MODIFICADA para buscar o histórico DO BANCO DE DADOS
# Esta é a função "ponte" entre o LangChain e o nosso MySQL


def get_session_history(session_id):
    # Se não há ID (é um novo chat), retorna histórico vazio
    if session_id is None:
        return ChatMessageHistory()
    mensagens_do_banco = carregar_mensagens(session_id)

    # Cria um objeto ChatMessageHistory em memória
    history = ChatMessageHistory()
    # Adiciona as mensagens carregadas (HumanMessage, AIMessage) ao objeto
    for msg in mensagens_do_banco:
        history.add_message(msg)
    return history


# Cria o template do prompt principal (para a conversa)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente prestativo. Responda às perguntas do usuário da forma mais completa e educada possível."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# Cria a "Chain" principal, agora com memória persistente
try:
    # RunnableWithMessageHistory é o componente mágico do LangChain que gerencia a memória
    chain_with_memory = RunnableWithMessageHistory(
        prompt_template | llm,      # 1. A chain principal (Prompt + LLM)
        get_session_history,      # 2. A *função* que ele deve usar para buscar/salvar o histórico
        input_messages_key="input",   # 3. O nome da variável de entrada do usuário no prompt
        history_messages_key="history",  # 4. O nome da variável de histórico no prompt
    )
except Exception as e:
    st.error(f"Erro Chain Memória: {e}")
    st.stop()

# --- 3. CONFIGURAÇÃO DO FRONTEND (Streamlit com Sidebar e Estado) ---
st.set_page_config(page_title="Chatbot com MySQL v4", layout="wide")
st.title("Meu Chatbot com Gemini e MySQL v4 💾🤖")

# --- Barra Lateral (Sidebar) para Gerenciar Conversas ---
st.sidebar.title("Minhas Conversas")

if st.sidebar.button("➕ Novo Chat", key="novo_chat_sidebar_button"):
    st.session_state.conversa_ativa_id = None
    # Força o recarregamento da página
    st.rerun()

# Listar conversas existentes do banco
try:
    lista_de_conversas = listar_conversas()
except Exception as e:
    st.sidebar.error(f"Erro ao listar conversas: {e}")
    lista_de_conversas = []

# st.session_state é a "memória RAM" do Streamlit.
# Usamos para guardar qual conversa está ativa e qual está sendo editada.

# Garante que temos um estado para a conversa ativa
if "conversa_ativa_id" not in st.session_state:
    st.session_state.conversa_ativa_id = None

# Garante que temos um estado para o "modo de edição"
if "editing_chat_id" not in st.session_state:
    # None significa que não estamos editando nada
    st.session_state.editing_chat_id = None

st.sidebar.divider()
st.sidebar.markdown("**Histórico:**")
if not lista_de_conversas:
    st.sidebar.info("Nenhuma conversa ainda.")
else:
    conversations_container = st.sidebar.container(height=300)
    with conversations_container:
        for conversa in lista_de_conversas:
            conversa_id = conversa['id']
            titulo_display = conversa.get(
                'titulo') or f'Conversa ID {conversa_id}'
            # --- LÓGICA DE EDIÇÃO ---
            # Verifica se esta conversa é a que está sendo editada
            if st.session_state.editing_chat_id == conversa_id:
                # Garante que temos um estado para o input de texto
                if f"edit_input_{conversa_id}" not in st.session_state:
                    st.session_state[f"edit_input_{conversa_id}"] = None

                # Mostra o input de texto com o título atual
                novo_titulo_input = st.text_input(
                    "Novo Título:",
                    value=titulo_display,
                    key=f"edit_input_{conversa_id}",
                    help="Pressione Enter ou clique em Salvar"
                )

                # Colunas para os botões Salvar e Cancelar
                col_salvar, col_cancelar = st.columns(2, gap="small")

                with col_salvar:
                    if st.button("Salvar", key=f"save_{conversa_id}", use_container_width=True, type="primary"):
                        if novo_titulo_input and novo_titulo_input != titulo_display:
                            # Chama a função do db.py para salvar no banco
                            if atualizar_titulo_conversa(conversa_id, novo_titulo_input):
                                st.toast("Título atualizado!", icon="✅")
                            else:
                                st.error("Erro ao salvar o título.")
                        st.session_state.editing_chat_id = None  # Sai do modo de edição
                        st.rerun()

                with col_cancelar:
                    if st.button("Cancelar", key=f"cancel_{conversa_id}", use_container_width=True):
                        st.session_state.editing_chat_id = None  # Sai do modo de edição
                        st.rerun()

            else:
                # --- MODO DE VISUALIZAÇÃO (Normal) ---
                # Cria TRÊS colunas: Título, Editar (✏️), Deletar (🗑️)
                col1, col2, col3 = st.columns([0.7, 0.15, 0.15], gap="small")

                with col1:
                    # Botão para selecionar a conversa
                    if st.button(titulo_display, key=f"conversa_{conversa_id}", use_container_width=True):
                        st.session_state.conversa_ativa_id = conversa_id
                        # Garante que sai de qualquer outro modo de edição
                        st.session_state.editing_chat_id = None
                        st.rerun()

                with col2:
                    # Botão de Editar ✏️
                    if st.button("✏️", key=f"edit_{conversa_id}", help="Renomear conversa", use_container_width=True):
                        st.session_state.editing_chat_id = conversa_id  # Entra em modo de edição
                        st.rerun()  # Recarrega para mostrar o input de texto

                with col3:
                    # Botão de Deletar 🗑️ (código que você já tinha)
                    if st.button("🗑️", key=f"delete_{conversa_id}", help=f"Deletar conversa {conversa_id}", use_container_width=True):
                        print(
                            f"DEBUG: Botão DELETAR conversa {conversa_id} clicado.")
                        try:
                            if deletar_conversa(conversa_id):
                                st.toast(
                                    f"Conversa {conversa_id} deletada.", icon="✅")
                                if st.session_state.get("conversa_ativa_id") == conversa_id:
                                    st.session_state.conversa_ativa_id = None
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error(
                                    f"Erro ao deletar conversa {conversa_id}.")
                        except Exception as e:
                            st.error(f"Erro inesperado ao deletar: {e}")


# --- Área Principal ---
# Pega o ID da conversa ativa (pode ser None se for um novo chat)
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
                st.rerun()  # Recarrega para mostrar a resposta salva
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
            # Define como ativa ANTES de salvar/invocar
            st.session_state.conversa_ativa_id = novo_id
            print(
                f"DEBUG: Novo chat criado (ID:{novo_id}). Mensagem: {prompt}")
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
                                titulo_response = chain_gerar_titulo.invoke(
                                    {"input": prompt})
                            if titulo_response and hasattr(titulo_response, 'content') and titulo_response.content.strip():
                                novo_titulo = titulo_response.content
                                if not atualizar_titulo_conversa(novo_id, novo_titulo):
                                    st.warning(
                                        "Não foi possível salvar o título gerado.")
                            else:
                                st.warning("LLM não gerou um título válido.")
                        except Exception as e_titulo:
                            st.warning(f"Erro ao gerar título: {e_titulo}")

                    print("DEBUG: Recarregando após novo chat.")
                    time.sleep(0.5)
                    st.rerun()  # Recarrega para mostrar tudo
                else:
                    st.warning(
                        "LLM retornou resposta vazia na primeira mensagem.")
            except Exception as e:
                st.error(f"Erro ao processar primeira mensagem: {e}")
        else:
            st.error("Falha ao criar nova conversa no banco.")
