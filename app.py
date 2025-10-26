import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
# Importações CORRETAS para a memória na versão > 1.0
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory # Para a memória em RAM
import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente (sua chave GEMINI_API_KEY) do arquivo .env
load_dotenv()

# --- 1. CONFIGURAÇÃO DO LLM (Conexão LLM - 30 min) ---
# Inicializa o modelo Gemini (sem alterações aqui)
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-preview-09-2025",
                             google_api_key=os.getenv("GEMINI_API_KEY"),
                             convert_system_message_to_human=True)


# --- 2. CONFIGURAÇÃO DO BACKEND (Memória Temporária - Modo Moderno) ---

# Função para buscar o histórico da memória (usando st.session_state)
# Esta função é exigida pelo RunnableWithMessageHistory
def get_session_history(session_id):
    if session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[session_id] = ChatMessageHistory()
    return st.session_state.chat_histories[session_id]

# Inicializa o armazenamento de históricos no session_state, se não existir
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

# Cria o template do prompt, incluindo um placeholder para o histórico
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente prestativo. Responda às perguntas do usuário."),
        MessagesPlaceholder(variable_name="history"), # Onde o histórico será inserido
        ("human", "{input}"), # Onde a pergunta do usuário será inserida
    ]
)

# Cria a "Chain" moderna com memória
# Esta chain agora sabe como buscar o histórico usando get_session_history
chain_with_memory = RunnableWithMessageHistory(
    prompt_template | llm, # Conecta o prompt ao LLM
    get_session_history,   # Função que fornece o histórico correto
    input_messages_key="input", # Nome da variável da entrada do usuário no prompt
    history_messages_key="history", # Nome da variável do histórico no prompt
)


# --- 3. CONFIGURAÇÃO DO FRONTEND (Streamlit - 1 hora) ---
st.set_page_config(page_title="Chatbot Funcional v2", layout="wide")
st.title("Meu Chatbot com Gemini (v2) 🤖")

# Define um ID de sessão padrão (poderia ser o ID do usuário, etc.)
# Para este exemplo simples, usamos um ID fixo "chat_principal"
session_id = "chat_principal"

# Exibe o histórico da conversa (lendo da memória vinculada à session_id)
chat_history = get_session_history(session_id)
for message in chat_history.messages:
    role = "ai" if isinstance(message, AIMessage) else "human"
    with st.chat_message(role):
        st.markdown(message.content)

# Pega a nova mensagem do usuário
if prompt := st.chat_input("Digite sua mensagem..."):
    
    # 1. (Frontend) Mostra a mensagem do usuário na tela
    with st.chat_message("human"):
        st.markdown(prompt)

    # 2. (Backend/LLM) Pensa na resposta E SALVA O HISTÓRICO
    # Usamos "invoke" na chain moderna, passando o input e a config (session_id)
    # A chain_with_memory automaticamente busca o histórico, chama o LLM e salva a nova resposta
    response = chain_with_memory.invoke(
        {"input": prompt},
        config={"configurable": {"session_id": session_id}} 
    )

    # 3. (Frontend) Mostra a resposta da IA na tela
    # A resposta agora está dentro do objeto response, no atributo 'content'
    with st.chat_message("ai"):
        st.markdown(response.content)
