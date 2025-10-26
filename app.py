import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
# ChatMessageHistory ainda é útil para *estruturar* a memória antes de passá-la para a chain
from langchain_community.chat_message_histories import ChatMessageHistory 
import os
from dotenv import load_dotenv
import time # Para adicionar um pequeno delay e melhorar a percepção

# Importa as funções do nosso backend de banco de dados
from db import (
    # criar_tabelas, # Não precisa chamar sempre, o db.py já faz isso se rodado diretamente
    listar_conversas, 
    criar_nova_conversa, 
    carregar_mensagens, 
    salvar_mensagem
)

# Carrega as variáveis de ambiente (.env)
load_dotenv()

# --- 1. CONFIGURAÇÃO DO LLM (Sem mudanças) ---
try:
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-preview-09-2025", # Usando o nome que funcionou
                                 google_api_key=os.getenv("GEMINI_API_KEY"),
                                 convert_system_message_to_human=True)
except Exception as e:
    st.error(f"Erro ao inicializar o LLM: {e}. Verifique sua chave de API e conexão.")
    st.stop() # Interrompe a execução se o LLM não puder ser carregado

# --- 2. CONFIGURAÇÃO DO BACKEND (Agora com Memória do Banco) ---

# Função MODIFICADA para buscar o histórico DO BANCO DE DADOS
# Esta função é crucial para o RunnableWithMessageHistory saber de onde ler/escrever
# session_id aqui será o ID da conversa no banco
def get_session_history(session_id):
    if session_id is None: # Se não há ID (novo chat), retorna histórico vazio
        # print("DEBUG: get_session_history chamado com session_id None. Retornando histórico vazio.")
        return ChatMessageHistory() 
        
    # Usa o session_id (que será o ID da conversa no banco) para carregar
    # Retorna uma lista de HumanMessage ou AIMessage
    # print(f"DEBUG: get_session_history tentando carregar mensagens para id_conversa: {session_id}")
    mensagens_do_banco = carregar_mensagens(session_id) 
    
    # Cria um objeto ChatMessageHistory em memória e popula com as mensagens do banco
    # A chain espera receber um objeto deste tipo
    history = ChatMessageHistory()
    for msg in mensagens_do_banco:
        history.add_message(msg)
    # print(f"DEBUG: Histórico carregado para session_id {session_id}: {len(history.messages)} mensagens.")
    return history

# NÃO precisamos mais inicializar st.session_state.chat_histories aqui
# A memória agora vive no banco de dados

# Cria o template do prompt (Sem mudanças)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente prestativo. Responda às perguntas do usuário da forma mais completa e educada possível."),
        MessagesPlaceholder(variable_name="history"), 
        ("human", "{input}"), 
    ]
)

# Cria a "Chain" moderna com memória (Sem mudanças na estrutura)
# Agora, get_session_history vai buscar no banco!
try:
    chain_with_memory = RunnableWithMessageHistory(
        prompt_template | llm, 
        get_session_history,    
        input_messages_key="input", 
        history_messages_key="history", 
    )
except Exception as e:
    st.error(f"Erro ao criar a chain com memória: {e}")
    st.stop()

# --- 3. CONFIGURAÇÃO DO FRONTEND (Streamlit com Sidebar e Estado) ---
st.set_page_config(page_title="Chatbot com MySQL", layout="wide")
st.title("Meu Chatbot com Gemini e MySQL 💾🤖")

# --- Barra Lateral (Sidebar) para Gerenciar Conversas ---
st.sidebar.title("Minhas Conversas")

# Botão para iniciar um novo chat
if st.sidebar.button("➕ Novo Chat", key="novo_chat_sidebar_button"):
    print("DEBUG: Botão Novo Chat clicado.")
    # Limpa o ID da conversa ativa para indicar que queremos um novo chat
    st.session_state.conversa_ativa_id = None
    # Força o recarregamento da página para limpar a tela principal
    st.rerun() 

# Listar conversas existentes do banco
try:
    # print("DEBUG: Listando conversas do banco...")
    lista_de_conversas = listar_conversas() 
    # print(f"DEBUG: Conversas encontradas: {lista_de_conversas}")
except Exception as e:
    st.sidebar.error(f"Erro ao listar conversas: {e}")
    lista_de_conversas = []

# Garante que temos um estado para a conversa ativa
# Se não houver nenhum ID salvo, começa como None
if "conversa_ativa_id" not in st.session_state:
    st.session_state.conversa_ativa_id = None 
    # print("DEBUG: st.session_state.conversa_ativa_id inicializado como None.")

# Mostra botões para cada conversa existente
st.sidebar.divider() # Adiciona uma linha divisória
st.sidebar.markdown("**Histórico:**")
if not lista_de_conversas:
    st.sidebar.info("Nenhuma conversa ainda.")
else:
    # Mostra primeiro as conversas mais recentes (já ordenado em listar_conversas)
    for conversa in lista_de_conversas: 
        # Usa o ID da conversa como chave única para o botão
        # Isso garante que o Streamlit saiba qual botão foi clicado
        # Usamos .get() para segurança caso 'titulo' não exista
        titulo_display = conversa.get('titulo', f'Conversa ID {conversa["id"]}') 
        if st.sidebar.button(titulo_display, key=f"conversa_{conversa['id']}", use_container_width=True):
            print(f"DEBUG: Botão da conversa {conversa['id']} clicado.")
            # Ao clicar, define este como o ID da conversa ativa
            st.session_state.conversa_ativa_id = conversa['id']
            # Recarrega a página para mostrar o histórico da conversa selecionada
            st.rerun() 

# --- Área Principal do Chat ---

# Verifica se há uma conversa ativa selecionada
active_chat_id = st.session_state.get("conversa_ativa_id") # Usar .get() é mais seguro
# print(f"DEBUG: ID da conversa ativa no início da renderização: {active_chat_id}")

if active_chat_id:
    
    # Mostra um título indicando qual conversa está ativa (opcional)
    # st.subheader(f"Conversa ID: {active_chat_id}")
    
    # Carrega o histórico da conversa ativa do banco para EXIBIÇÃO
    # A chain usa get_session_history internamente, mas precisamos buscar de novo 
    # para mostrar as mensagens na tela antes do novo input.
    try:
        # print(f"DEBUG: Carregando histórico para exibição (ID: {active_chat_id})")
        chat_history_para_exibir = get_session_history(active_chat_id)
        # print(f"DEBUG: Histórico para exibição carregado: {len(chat_history_para_exibir.messages)} mensagens.")
    except Exception as e:
        st.error(f"Erro ao carregar histórico para exibição: {e}")
        chat_history_para_exibir = ChatMessageHistory() # Mostra vazio em caso de erro

    # Exibe o histórico carregado
    for message in chat_history_para_exibir.messages:
        role = "ai" if isinstance(message, AIMessage) else "human"
        with st.chat_message(role):
            st.markdown(message.content)

    # Pega a nova mensagem do usuário
    if prompt := st.chat_input("Digite sua mensagem..."):
        print(f"DEBUG: Usuário digitou: {prompt}")
        
        # 1. (Frontend) Mostra a mensagem do usuário na tela
        with st.chat_message("human"):
            st.markdown(prompt)
        
        # Salva a mensagem humana no banco PRIMEIRO
        if not salvar_mensagem(active_chat_id, "human", prompt):
             st.error("Erro ao salvar sua mensagem no banco de dados.")
             st.stop() # Interrompe se não conseguir salvar a mensagem do usuário
        # else: # Só continua se salvou com sucesso 
        
        # 2. (Backend/LLM) Pensa na resposta 
        # A chain usa get_session_history para pegar o histórico atualizado (incluindo o prompt acima)
        try:
            # Exibe um spinner enquanto o LLM pensa
            with st.spinner("Digitando..."):
                print(f"DEBUG: Chamando chain.invoke para conversa {active_chat_id}")
                response = chain_with_memory.invoke(
                    {"input": prompt},
                    # Passa o ID da conversa para a chain saber qual histórico usar/atualizar
                    config={"configurable": {"session_id": active_chat_id}} 
                )
                print(f"DEBUG: Resposta recebida do LLM: {response}")

            # 3. (Frontend) Mostra a resposta da IA na tela (apenas se não houve erro)
            if response and hasattr(response, 'content'):
                with st.chat_message("ai"):
                    st.markdown(response.content)
                    
                # Salva a mensagem da IA no banco DEPOIS que ela foi recebida e exibida
                if not salvar_mensagem(active_chat_id, "ai", response.content):
                      st.error("Erro ao salvar a resposta da IA no banco de dados.")
                
                # Não precisa recarregar aqui, o Streamlit atualiza a tela
                # st.rerun() # Remover ou comentar esta linha

            else:
                st.error("O LLM não retornou uma resposta válida.")


        except Exception as e:
            st.error(f"Erro ao chamar o LLM ou salvar a resposta: {e}")
            # Considerar não fazer st.rerun() aqui para o usuário ver o erro

else:
    # Se nenhuma conversa está selecionada, mostra uma mensagem inicial
    st.info("⬅️ Selecione uma conversa na barra lateral ou clique em '➕ Novo Chat' para começar.")
    
    # Caixa de input para iniciar um NOVO chat
    if prompt := st.chat_input("Digite sua primeira mensagem para iniciar um novo chat...", key="novo_chat_input"):
         print(f"DEBUG: Usuário digitou a primeira mensagem: {prompt}")
         # Cria uma nova conversa no banco ANTES de enviar a primeira mensagem
         try:
             print("DEBUG: Tentando criar nova conversa no banco...")
             novo_id = criar_nova_conversa()
         except Exception as e:
             st.error(f"Erro ao criar nova conversa no banco: {e}")
             novo_id = None
             
         if novo_id:
             st.session_state.conversa_ativa_id = novo_id # Define como ativa
             print(f"DEBUG: Nova conversa criada com ID {novo_id}. Definido como ativa.")
             
             # Salva a primeira mensagem humana
             if not salvar_mensagem(novo_id, "human", prompt):
                  st.error("Erro ao salvar sua primeira mensagem.")
             # else: # Só continua se salvou
             # Pensa na resposta (a chain vai carregar o histórico vazio + a 1a msg humana)
             try:
                 # Exibe um spinner enquanto o LLM pensa
                 with st.spinner("Digitando..."):
                     print(f"DEBUG: Chamando chain.invoke pela primeira vez para conversa {novo_id}")
                     response = chain_with_memory.invoke(
                         {"input": prompt},
                         config={"configurable": {"session_id": novo_id}} 
                     )
                     print(f"DEBUG: Primeira resposta recebida do LLM: {response}")
                 
                 # Salva a primeira resposta da IA (apenas se válida)
                 if response and hasattr(response, 'content'):
                     if not salvar_mensagem(novo_id, "ai", response.content):
                           st.error("Erro ao salvar a primeira resposta da IA.")
                     
                     # Recarrega a página para mostrar o novo chat com as mensagens
                     # e atualizar a sidebar
                     print("DEBUG: Recarregando a página (st.rerun) após a primeira interação.")
                     time.sleep(0.5) # Pequeno delay para garantir que o salvamento no DB conclua
                     st.rerun() 
                 else:
                      st.error("O LLM não retornou uma resposta válida para a primeira mensagem.")

             except Exception as e:
                  st.error(f"Erro ao chamar o LLM para a primeira mensagem: {e}")
         else:
             st.error("Não foi possível criar uma nova conversa no banco.")
