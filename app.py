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

# --- IMPORTAÇÕES PARA O AGENTE SQL (ADICIONADAS) ---
from langchain_core.tools import tool
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
# --- FIM DAS NOVAS IMPORTAÇÕES ---

# Importa as funções do db
from db import (
    db_engine,  # <-- IMPORTA A ENGINE SQLALCHEMY
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
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-preview-09-2025", # <--- SEU MODELO ORIGINAL (O CORRETO)
                                 google_api_key=os.getenv("GEMINI_API_KEY"),
                                 convert_system_message_to_human=True)
except Exception as e:
    st.error(f"Erro LLM: {e}")
    st.stop()

# --- 1.1 Mini-Chain para gerar títulos ---
# (Exatamente como no seu v4)
try:
    prompt_titulo_template = ChatPromptTemplate.from_template(
        "Gere um título muito curto e descritivo (máximo 5 palavras, idealmente 2-3) para uma conversa de chatbot que começa com a seguinte mensagem do usuário: '{primeira_mensagem}'. O título deve resumir o tópico principal. Responda APENAS com o título, sem introduções como 'Título:', sem aspas e sem pontuação final."
    )
    chain_gerar_titulo = RunnablePassthrough.assign(
        primeira_mensagem=lambda x: x['input']) | prompt_titulo_template | llm
    print("DEBUG: Chain de título criada.")
except Exception as e:
    st.warning(f"Aviso: Chain de título não criada: {e}")
    chain_gerar_titulo = None

# --- 2. CONFIGURAÇÃO DO BACKEND (DOIS "CÉREBROS") ---

# --- CÉREBRO 1: CHAT GERAL (Exatamente como no seu v4) ---
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
    print("DEBUG: Cérebro 1 (Chat Geral com Memória) criado.")
except Exception as e:
    st.error(f"Erro Chain Memória: {e}")
    st.stop()

# --- CÉREBRO 2: AGENTE DE VENDAS (Lógica do v5) ---
agente_sql = None
especialista_vendas = None
try:
    if db_engine is None:
        st.warning("Aviso: Engine SQLAlchemy não foi criada. O 'Modo Vendas' não funcionará.")
    else:
        # 1. Conecta o LangChain ao banco (usando a engine do db.py)
        db_sql = SQLDatabase(engine=db_engine, include_tables=['vendas'])

        # 2. Cria o Agente SQL (um sub-agente focado apenas em SQL)
        agente_sql = create_sql_agent(
            llm=llm,
            db=db_sql,
            verbose=True, 
            agent_type="tool-calling"
        )
        
        # 3. "Embrulhamos" o Agente SQL em uma @tool 
        #    (Mesmo que não seja usada por outro agente, é uma boa prática)
        @tool
        def especialista_vendas(input: str): 
            """
            Responde perguntas APENAS sobre a tabela 'vendas'.
            """
            print(f"DEBUG: Cérebro 2 (Especialista Vendas) chamado com input: {input}")
            try:
                resultado = agente_sql.invoke({"input": input})
                return resultado.get("output", "Não consegui processar a consulta SQL.")
            except Exception as e:
                print(f"ERRO no especialista_vendas: {e}")
                return f"Houve um erro ao consultar o banco de dados de vendas: {e}"
        
        print("DEBUG: Cérebro 2 (Especialista Vendas) criado com sucesso.")

except Exception as e:
    st.error(f"ERRO CRÍTICO: Não foi possível criar o Agente SQL: {e}")


# --- 3. CONFIGURAÇÃO DO FRONTEND (Streamlit) ---
st.set_page_config(page_title="Chatbot com MySQL v4 + Agente", layout="wide")
st.title("Meu Chatbot com Gemini e MySQL 💾🤖")

# --- Barra Lateral (Sidebar) ---
st.sidebar.title("Minhas Conversas")

# --- BOTÃO DE MODO (DE VOLTA!) ---
if "modo_vendas" not in st.session_state:
    st.session_state.modo_vendas = False

st.session_state.modo_vendas = st.sidebar.toggle(
    "Modo Consulta de Vendas", 
    key="modo_vendas_toggle",
    value=st.session_state.modo_vendas,
    help="Se ativado, o chat responderá APENAS sobre a tabela de Vendas. Se desativado, funcionará como um chat geral com memória."
)

if st.session_state.modo_vendas:
    st.sidebar.info("Modo Consulta de Vendas (SQL) ATIVADO.")
else:
    st.sidebar.info("Modo Conversa Geral ATIVADO.")
# --- FIM DO BOTÃO DE MODO ---


if st.sidebar.button("➕ Novo Chat", key="novo_chat_sidebar_button"):
    st.session_state.conversa_ativa_id = None
    st.rerun()

# --- Lógica da Barra Lateral (Listar, Editar, Deletar) ---
# (Exatamente como no seu v4, que já estava perfeito)
try:
    lista_de_conversas = listar_conversas()
except Exception as e:
    st.sidebar.error(f"Erro ao listar conversas: {e}")
    lista_de_conversas = []
if "conversa_ativa_id" not in st.session_state:
    st.session_state.conversa_ativa_id = None
if "editing_chat_id" not in st.session_state:
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
            titulo_display = conversa.get('titulo') or f'Conversa ID {conversa_id}'
            
            if st.session_state.editing_chat_id == conversa_id:
                if f"edit_input_{conversa_id}" not in st.session_state:
                    st.session_state[f"edit_input_{conversa_id}"] = None
                novo_titulo_input = st.text_input(
                    "Novo Título:", value=titulo_display, key=f"edit_input_{conversa_id}",
                    help="Pressione Enter ou clique em Salvar"
                )
                col_salvar, col_cancelar = st.columns(2, gap="small")
                with col_salvar:
                    if st.button("Salvar", key=f"save_{conversa_id}", use_container_width=True, type="primary"):
                        if novo_titulo_input and novo_titulo_input != titulo_display:
                            if atualizar_titulo_conversa(conversa_id, novo_titulo_input):
                                st.toast("Título atualizado!", icon="✅")
                            else:
                                st.error("Erro ao salvar o título.")
                        st.session_state.editing_chat_id = None
                        st.rerun()
                with col_cancelar:
                    if st.button("Cancelar", key=f"cancel_{conversa_id}", use_container_width=True):
                        st.session_state.editing_chat_id = None
                        st.rerun()
            else:
                col1, col2, col3 = st.columns([0.7, 0.15, 0.15], gap="small")
                with col1:
                    if st.button(titulo_display, key=f"conversa_{conversa_id}", use_container_width=True):
                        st.session_state.conversa_ativa_id = conversa_id
                        st.session_state.editing_chat_id = None
                        st.rerun()
                with col2:
                    if st.button("✏️", key=f"edit_{conversa_id}", help="Renomear conversa", use_container_width=True):
                        st.session_state.editing_chat_id = conversa_id
                        st.rerun()
                with col3:
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


# --- INPUT ÚNICO (COM A LÓGICA DO BOTÃO E SALVAMENTO CORRIGIDO) ---

# Define o placeholder dinamicamente baseado no modo
if st.session_state.get("modo_vendas"):
    placeholder = "Pergunte APENAS sobre os dados de Vendas (Salva no histórico)..."
elif active_chat_id:
    placeholder = "Digite sua mensagem..."
else:
    placeholder = "Digite sua primeira mensagem para iniciar um novo chat..."


if prompt := st.chat_input(placeholder, key="chat_input_principal"):

    # --- LÓGICA DO BOTÃO "Modo Vendas" (AGORA SALVANDO) ---
    if st.session_state.get("modo_vendas"):
        
        print(f"DEBUG: Modo Vendas. Pergunta: {prompt}")

        # 0. Verificar se o agente existe ANTES de tudo
        if not especialista_vendas:
            st.error("O Agente SQL não está disponível. Verifique os erros no terminal.")
            st.stop() # Para a execução se o modo vendas foi ligado mas o agente falhou

        # 1. Obter o chat_id ATUAL
        active_chat_id = st.session_state.get("conversa_ativa_id")
        is_new_chat = False

        # 2. LÓGICA DE NOVO CHAT (copiada do Modo Geral)
        #    Se não há chat ativo, cria um novo
        if active_chat_id is None:
            try:
                novo_id = criar_nova_conversa()
                if novo_id:
                    st.session_state.conversa_ativa_id = novo_id
                    active_chat_id = novo_id
                    is_new_chat = True
                    print(f"DEBUG: Novo chat (Modo Vendas) criado (ID:{novo_id}).")
                else:
                    st.error("Falha ao criar nova conversa no banco.")
                    st.stop()
            except Exception as e:
                st.error(f"Erro ao criar nova conversa: {e}")
                st.stop()

        # 3. Salvar a mensagem HUMANA (agora temos um active_chat_id)
        if not salvar_mensagem(active_chat_id, "human", prompt):
            st.error("Erro ao salvar sua mensagem.")
            st.stop()

        # 4. Chamar o CÉREBRO 2 (Agente SQL)
        try:
            with st.spinner("Consultando banco de dados de Vendas..."):
                resposta_final = especialista_vendas.invoke(prompt)

            # 5. Salvar a resposta da IA
            if resposta_final and resposta_final.strip():
                if not salvar_mensagem(active_chat_id, "ai", resposta_final):
                    st.error("Erro ao salvar a resposta da IA.")
                    st.stop()
            else:
                st.warning("O Agente SQL retornou uma resposta vazia.")

            # 6. Gerar Título (se for novo chat)
            #    (Lógica exata do Modo Geral)
            if is_new_chat and chain_gerar_titulo:
                try:
                    with st.spinner("Gerando título..."):
                        titulo_response = chain_gerar_titulo.invoke({"input": prompt})
                        if titulo_response and hasattr(titulo_response, 'content') and titulo_response.content.strip():
                            novo_titulo = titulo_response.content
                            if not atualizar_titulo_conversa(active_chat_id, novo_titulo):
                                st.warning("Não foi possível salvar o título gerado.")
                        else:
                            st.warning("LLM não gerou um título válido.")
                except Exception as e_titulo:
                    st.warning(f"Erro ao gerar título: {e_titulo}")

            # 7. Recarregar a página para mostrar as mensagens salvas
            time.sleep(0.1) 
            st.rerun()

        except Exception as e:
            st.error(f"Erro ao consultar o Agente SQL: {e}")
            print(f"ERRO DETALHADO DO AGENTE SQL: {e}")


    else:
        
        # --- CÉREBRO 1 (CHAT GERAL - CÓDIGO DO v4 INTACTO) ---
        print(f"DEBUG: Modo Chat Geral. Pergunta: {prompt}")

        # Lógica se JÁ EXISTE um chat ativo (do v4)
        if active_chat_id:
            # (O st.chat_message("human") é desnecessário aqui, 
            # pois o st.rerun() vai exibir o histórico salvo)
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
                    st.rerun()
                else:
                    st.warning("O LLM retornou uma resposta vazia.")
            except Exception as e:
                st.error(f"Erro ao processar mensagem: {e}")

        # Lógica se é um NOVO chat (do v4)
        else:
            try:
                novo_id = criar_nova_conversa()
            except Exception as e:
                st.error(f"Erro ao criar nova conversa: {e}")
                novo_id = None

            if novo_id:
                st.session_state.conversa_ativa_id = novo_id
                print(f"DEBUG: Novo chat (Modo Geral) criado (ID:{novo_id}). Mensagem: {prompt}")
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
                        st.rerun()
                    else:
                        st.warning("LLM retornou resposta vazia na primeira mensagem.")
                except Exception as e:
                    st.error(f"Erro ao processar primeira mensagem: {e}")
            else:
                st.error("Falha ao criar nova conversa no banco.")