import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
import os
from dotenv import load_dotenv
import time 

# --- IMPORTAÇÕES PARA O AGENTE SQL (Existentes) ---
from langchain_core.tools import tool
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase

# --- NOVAS IMPORTAÇÕES (RAG/Embeddings e PDF) ---
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

# Importa as funções do db (incluindo a db_engine)
from db import (
    db_engine, 
    listar_conversas,
    criar_nova_conversa,
    carregar_mensagens,
    salvar_mensagem,
    deletar_conversa,
    atualizar_titulo_conversa
)

# Carrega as variáveis de ambiente
load_dotenv()

# --- Configuração do LLM e Embeddings ---
try:
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-preview-09-2025",
                                 google_api_key=os.getenv("GEMINI_API_KEY"),
                                 convert_system_message_to_human=True)
    
    embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
    # Deixamos a biblioteca decidir o device (ela vai usar CPU)
    )
    print("DEBUG: Embeddings locais (HuggingFace) carregados.")
except Exception as e:
    st.error(f"Erro LLM/Embeddings: {e}")
    st.stop()

# --- 1.1 Mini-Chain para gerar títulos ---
# (Sem alterações, continua perfeita)
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

# --- 2. CONFIGURAÇÃO DOS 3 "CÉREBROS" ---

# Função para buscar o histórico DO BANCO DE DADOS (usada por todos)
def get_session_history(session_id):
    if session_id is None:
        return ChatMessageHistory()
    mensagens_do_banco = carregar_mensagens(session_id)
    history = ChatMessageHistory()
    for msg in mensagens_do_banco:
        history.add_message(msg)
    return history

# --- CÉREBRO 1: CHAT GERAL (RESTAURADO) ---
# Este é o seu 'v4' que funcionava
try:
    prompt_template_geral = ChatPromptTemplate.from_messages(
        [
            ("system", "Você é um assistente prestativo. Responda às perguntas do usuário da forma mais completa e educada possível."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    chain_with_memory = RunnableWithMessageHistory(
        prompt_template_geral | llm, 
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    print("DEBUG: Cérebro 1 (Chat Geral) RESTAURADO e criado.")
except Exception as e:
    st.error(f"Erro Chain Memória: {e}")
    st.stop()

# --- CÉREBRO 2: AGENTE DE VENDAS (SQL) ---
# (Sem alterações, continua perfeito)
agente_sql = None
especialista_vendas = None
try:
    if db_engine is None:
        st.warning("Aviso: Engine SQLAlchemy não foi criada. O 'Modo Vendas' não funcionará.")
    else:
        db_sql = SQLDatabase(engine=db_engine, include_tables=['vendas'])
        agente_sql_executor = create_sql_agent(
            llm=llm,
            db=db_sql,
            verbose=True, 
            agent_type="tool-calling" # Corrigido com hífen
        )
        
        def especialista_vendas(input_str: str): 
            print(f"DEBUG: Cérebro 2 (Especialista Vendas) chamado com input: {input_str}")
            try:
                resultado = agente_sql_executor.invoke({"input": input_str})
                return resultado.get("output", "Não consegui processar a consulta SQL.")
            except Exception as e:
                print(f"ERRO no especialista_vendas: {e}")
                return f"Houve um erro ao consultar o banco de dados de vendas: {e}"
        
        print("DEBUG: Cérebro 2 (Especialista Vendas) criado com sucesso.")

except Exception as e:
    st.error(f"ERRO CRÍTICO: Não foi possível criar o Agente SQL: {e}")

# --- CÉREBRO 3: CONSULTOR DE DOCUMENTOS (RAG) ---

# A OTIMIZAÇÃO: @st.cache_resource
@st.cache_resource(ttl=3600) # Limpa o cache a cada 1 hora
def processar_pdf_para_rag(_file_id, file_content, file_name):
    """
    Processa o PDF anexado e RETORNA a chain RAG pronta.
    O @st.cache_resource impede que isso rode duas vezes para o mesmo arquivo.
    """
    print(f"DEBUG: Processando PDF '{file_name}' PELA PRIMEIRA VEZ (Gastando Quota de API)...")
    try:
        # Salva o arquivo temporariamente
        with open(file_name, "wb") as f:
            f.write(file_content)
        
        loader = PyPDFLoader(file_name)
        docs = loader.load()
        os.remove(file_name) # Limpa o arquivo temporário

        if not docs:
            print("Erro: Não foi possível ler o conteúdo do PDF.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # ATENÇÃO: É AQUI QUE O SEU ERRO 429 (QUOTA) VAI ACONTECER (NA 1ª VEZ)
        vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        retriever = vector_store.as_retriever()
        
        rag_prompt = ChatPromptTemplate.from_template(
            """Baseado APENAS no contexto abaixo, responda à pergunta:
            Contexto: {contexto}
            Pergunta: {pergunta}
            Resposta:"""
        )
        
        rag_chain = (
            RunnablePassthrough.assign(contexto=(lambda x: retriever.invoke(x["pergunta"])))
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        
        print(f"DEBUG: PDF '{file_name}' processado e 'chain' criada com sucesso.")
        return rag_chain
        
    except Exception as e:
        # O ERRO 429 VAI APARECER AQUI
        print(f"ERRO DETALHADO ao processar PDF: {e}")
        if os.path.exists(file_name):
            os.remove(file_name)
        # Re-lança o erro para o Streamlit mostrar
        raise e


# --- CÉREBRO 0: O ROTEADOR (NOVO) ---
try:
    roteador_prompt_template = """
    Sua tarefa é classificar a pergunta do usuário em uma de três categorias: 'SQL', 'RAG', ou 'GERAL'.

    Contexto:
    - Um PDF está anexado: {contexto_rag}

    Regras de Classificação:
    1.  Se a pergunta for sobre vendas, clientes, produtos, valores, faturamento, ou qualquer coisa da tabela 'vendas', 
        responda APENAS com a palavra: SQL
    2.  Se {contexto_rag} for True E a pergunta for sobre o documento PDF anexado (como garantias, políticas, termos, etc.), 
        responda APENAS com a palavra: RAG
    3.  Para todo o resto (cumprimentos, piadas, conversas aleatórias, ou se {contexto_rag} for False e a pergunta for sobre um PDF), 
        responda APENAS com a palavra: GERAL

    Pergunta do Usuário:
    '{input}'
    """
    
    roteador_prompt = ChatPromptTemplate.from_template(roteador_prompt_template)
    chain_roteadora = roteador_prompt | llm | StrOutputParser()
    print("DEBUG: Cérebro 0 (Roteador) criado com sucesso.")
except Exception as e:
    st.error(f"Erro ao criar o Roteador: {e}")
    st.stop()


# --- 3. CONFIGURAÇÃO DO FRONTEND (Streamlit) ---
st.set_page_config(page_title="Chatbot Roteador (SQL/RAG/Geral)", layout="wide")
st.title("Meu Chatbot com Gemini (Roteador Automático) 💾🤖")

# --- Barra Lateral (Sidebar) ---
st.sidebar.title("Minhas Conversas")

if st.sidebar.button("➕ Novo Chat", key="novo_chat_sidebar_button"):
    st.session_state.clear() # Limpa TUDO (ID do chat, RAG, etc)
    st.rerun()

st.sidebar.divider()

# --- CORREÇÃO: UPLOADER DE VOLTA À SIDEBAR (Seu Pedido) ---
uploaded_file = st.sidebar.file_uploader(
    "Anexe um PDF para fazer perguntas sobre ele", 
    type="pdf", 
    key="sidebar_uploader"
)

if uploaded_file:
    # Checa se o arquivo é novo
    if "rag_file_name" not in st.session_state or st.session_state.rag_file_name != uploaded_file.name:
        try:
            file_id = uploaded_file.file_id
            file_content = uploaded_file.getvalue()
            file_name = uploaded_file.name
            
            # Tenta processar (só vai gastar API na 1ª vez)
            rag_chain = processar_pdf_para_rag(file_id, file_content, file_name)
            
            if rag_chain:
                # Salva a chain e o nome na sessão
                st.session_state.rag_chain = rag_chain
                st.session_state.rag_file_name = file_name
                st.sidebar.success(f"'{file_name}' processado e pronto!")
                
                # Se for um chat novo, cria ele agora
                if "conversa_ativa_id" not in st.session_state or st.session_state.conversa_ativa_id is None:
                    active_chat_id = criar_nova_conversa(titulo=f"Chat sobre {file_name}")
                    st.session_state.conversa_ativa_id = active_chat_id
                    st.rerun() # Recarrega para o novo chat aparecer
                
                # Salva uma msg no histórico do chat ATIVO
                salvar_mensagem(st.session_state.conversa_ativa_id, "ai", f"Certo! Estou pronto para responder perguntas sobre o documento '{file_name}'.")
                st.rerun() 
            else:
                st.sidebar.error("Falha ao processar o PDF.")

        except Exception as e:
            # O erro 429 vai aparecer aqui
            st.sidebar.error(f"Falha ao processar o PDF. (Erro 429?)")
elif "rag_file_name" in st.session_state:
    # Se já existe um PDF, mostra que ele está ativo
    st.sidebar.info(f"Contexto do PDF '{st.session_state.rag_file_name}' está ativo.")

st.sidebar.divider()
# --- FIM DO UPLOADER ---

# --- Lógica da Barra Lateral (Listar, Editar, Deletar) ---
try:
    lista_de_conversas = listar_conversas()
except Exception as e:
    st.sidebar.error(f"Erro ao listar conversas: {e}")
    lista_de_conversas = []
if "conversa_ativa_id" not in st.session_state:
    st.session_state.conversa_ativa_id = None
if "editing_chat_id" not in st.session_state:
    st.session_state.editing_chat_id = None

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
                # ... (lógica de edição) ...
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
                # ... (lógica de visualização) ...
                col1, col2, col3 = st.columns([0.7, 0.15, 0.15], gap="small")
                with col1:
                    if st.button(titulo_display, key=f"conversa_{conversa_id}", use_container_width=True):
                        st.session_state.conversa_ativa_id = conversa_id
                        st.session_state.editing_chat_id = None
                        
                        # --- CORREÇÃO: "PDF GLOBAL" (Seu Pedido) ---
                        # As linhas que limpavam o RAG foram removidas daqui.
                        # Agora o PDF persiste entre as trocas de chat.
                        
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
    if "rag_file_name" not in st.session_state:
         st.info("⬅️ Selecione uma conversa, anexe um PDF, ou digite abaixo para iniciar um novo chat.")

# --- LÓGICA DE UPLOAD (REMOVIDA DA ÁREA PRINCIPAL) ---

# --- INPUT ÚNICO (LÓGICA DOS 3 CÉREBROS + ROTEADOR) ---

placeholder = "Pergunte sobre vendas, o PDF anexado, ou apenas converse..."
if "rag_chain" in st.session_state:
    placeholder = f"Pergunte sobre '{st.session_state.rag_file_name}'..."
elif not active_chat_id:
    placeholder = "Digite sua primeira mensagem para iniciar um novo chat..."


if prompt := st.chat_input(placeholder, key="chat_input_principal"):

    # 1. Obter o chat_id ATUAL ou CRIAR UM NOVO
    is_new_chat = False
    if active_chat_id is None:
        try:
            novo_id = criar_nova_conversa()
            if novo_id:
                st.session_state.conversa_ativa_id = novo_id
                active_chat_id = novo_id
                is_new_chat = True
                print(f"DEBUG: Novo chat (ID:{novo_id}).")
            else:
                st.error("Falha ao criar nova conversa no banco.")
                st.stop()
        except Exception as e:
            st.error(f"Erro ao criar nova conversa: {e}")
            st.stop()

    # 2. Salvar a mensagem HUMANA
    if not salvar_mensagem(active_chat_id, "human", prompt):
        st.error("Erro ao salvar sua mensagem.")
        st.stop()
    
    # 3. Chamar o ROTEADOR (Cérebro 0) para decidir
    response_content = ""
    try:
        rag_anexado = "rag_chain" in st.session_state
        with st.spinner("Analisando sua pergunta..."):
            categoria = chain_roteadora.invoke({
                "input": prompt,
                "contexto_rag": rag_anexado
            })
        print(f"DEBUG: Roteador decidiu -> {categoria}")

        # 4. Executar o "Cérebro" correto com base na decisão
        
        # --- CÉREBRO 3 (RAG) ---
        if "RAG" in categoria:
            print(f"DEBUG: Modo RAG. Pergunta: {prompt}")
            rag_chain = st.session_state.rag_chain
            with st.spinner(f"Consultando '{st.session_state.rag_file_name}'..."):
                response_content = rag_chain.invoke({"pergunta": prompt})

        # --- CÉREBRO 2 (SQL) ---
        elif "SQL" in categoria:
            print(f"DEBUG: Modo Vendas. Pergunta: {prompt}")
            if not especialista_vendas:
                st.error("O Agente SQL não está disponível. Verifique os erros no terminal.")
                st.stop()
            with st.spinner("Consultando banco de dados de Vendas..."):
                response_content = especialista_vendas(prompt) # Chama a função direto

        # --- CÉREBRO 1 (CHAT GERAL) ---
        else: # Categoria "GERAL"
            print(f"DEBUG: Modo Chat Geral. Pergunta: {prompt}")
            with st.spinner("Digitando..."):
                response = chain_with_memory.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": active_chat_id}}
                )
                response_content = response.content if hasattr(response, 'content') else str(response)

        # 5. Salvar a resposta da IA
        if response_content and response_content.strip():
            if not salvar_mensagem(active_chat_id, "ai", response_content):
                st.error("Erro ao salvar a resposta da IA.")
                st.stop()
        else:
            st.warning("O LLM retornou uma resposta vazia.")

        # 6. Gerar Título (se for novo) - LÓGICA CORRIGIDA!
        if is_new_chat and chain_gerar_titulo:
            print("DEBUG: Novo chat, tentando gerar título...")
            try: # <-- O 'try' que estava causando o erro
                with st.spinner("Gerando título..."):
                    titulo_response = chain_gerar_titulo.invoke({"input": prompt})
                    if titulo_response and hasattr(titulo_response, 'content') and titulo_response.content.strip():
                        novo_titulo = titulo_response.content
                        print(f"DEBUG: Título gerado: {novo_titulo}")
                        if not atualizar_titulo_conversa(active_chat_id, novo_titulo):
                            st.warning("Não foi possível salvar o título gerado.")
                    else:
                        st.warning("LLM não gerou um título válido.")
            except Exception as e_titulo: # <-- O 'except' QUE FALTAVA
                st.warning(f"Erro ao gerar título: {e_titulo}")

        # 7. Recarregar a página para mostrar as mensagens salvas
        time.sleep(0.1) 
        st.rerun()

    except Exception as e:
        # O erro 429 (Quota) do Google aparecerá aqui se o RAG for usado
        st.error(f"Erro ao processar mensagem: {e}")
        print(f"ERRO DETALHADO NO PROCESSAMENTO: {e}")