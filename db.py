import mysql.connector
import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

# Carrega as variáveis de ambiente (DB_HOST, DB_USER, etc.) do arquivo .env
load_dotenv()

# LINHA DE TESTE - Adicione isto
print(
    f"DEBUG: Valor lido para DB_PORT do .env: '{os.getenv('DB_PORT', 'NaoDefinido')}'")

# --- Configurações de Conexão ---

# Limpa o DB_PORT (limpa o # e as aspas)
port_str = os.getenv("DB_PORT", "3306").strip().split('#')[
    0].strip().strip('"')
print(f"DEBUG: Valor APÓS limpeza (port_str): '{port_str}'")
port_int = int(port_str)

# Limpa o DB_HOST
host_str = os.getenv("DB_HOST", "localhost").strip().split('#')[
    0].strip().strip('"')
print(
    f"DEBUG: Valor lido para DB_HOST do .env: '{os.getenv('DB_HOST', 'NaoDefinido')}'")
print(f"DEBUG: Valor APÓS limpeza (host_str): '{host_str}'")

# --- ADICIONE ESTA LIMPEZA PARA O USER ---
user_str = os.getenv("DB_USER", "root").strip().split('#')[
    0].strip().strip('"')
print(
    f"DEBUG: Valor lido para DB_USER do .env: '{os.getenv('DB_USER', 'NaoDefinido')}'")
print(f"DEBUG: Valor APÓS limpeza (user_str): '{user_str}'")
# -----------------------------------------

# --- ADICIONE ESTA LIMPEZA PARA A SENHA ---
password_str = os.getenv("DB_PASSWORD", "").strip().split(
    '#')[0].strip().strip('"')  # Usa "" como padrão se não definida
print(
    f"DEBUG: Valor lido para DB_PASSWORD do .env: '{os.getenv('DB_PASSWORD', 'NaoDefinido')}'")
# CUIDADO: Não mostre a senha real em produção
print(f"DEBUG: Valor APÓS limpeza (password_str): '{password_str}'")
# -----------------------------------------

# --- ADICIONE ESTA LIMPEZA PARA O DATABASE NAME ---
database_str = os.getenv("DB_NAME", "projeto_chat").strip().split('#')[0].strip().strip('"').strip("'")
print(f"DEBUG: Valor lido para DB_NAME do .env: '{os.getenv('DB_NAME', 'NaoDefinido')}'") 
print(f"DEBUG: Valor APÓS limpeza (database_str): '{database_str}'") 
#-----------------------------------------

db_config = {
    'host': host_str,  # Usa a variável limpa
    'user': user_str,  # Usa a variável limpa
    'password': password_str,
    'database': database_str,
    'port': port_int
}

# --- Funções de Interação com o Banco ---


def get_db_connection():
    """Cria e retorna uma nova conexão com o banco."""
    try:
        conn = mysql.connector.connect(**db_config)
        # print("Conexão com MySQL bem-sucedida!") # Descomente para debug
        return conn
    except mysql.connector.Error as err:
        print(f"Erro ao conectar ao MySQL: {err}")
        # Em um app real, você poderia tentar reconectar ou levantar um erro no Streamlit
        return None


def criar_tabelas():
    """Cria as tabelas 'conversas' e 'mensagens' se elas não existirem."""
    conn = get_db_connection()
    if not conn:
        print("Não foi possível conectar ao banco para criar tabelas.")
        return

    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversas (
                id INT AUTO_INCREMENT PRIMARY KEY,
                titulo VARCHAR(255) DEFAULT 'Nova Conversa',
                data_criacao TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mensagens (
                id INT AUTO_INCREMENT PRIMARY KEY,
                id_conversa INT NOT NULL,
                role VARCHAR(10) NOT NULL CHECK (role IN ('human', 'ai')),
                content TEXT NOT NULL,
                data_envio TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (id_conversa) REFERENCES conversas(id) ON DELETE CASCADE
            );
        """)
        conn.commit()
        print("Tabelas verificadas/criadas com sucesso.")
    except mysql.connector.Error as err:
        print(f"Erro ao criar tabelas: {err}")
    finally:
        cursor.close()
        conn.close()


def listar_conversas():
    """Retorna uma lista de dicionários, cada um representando uma conversa (id, titulo)."""
    conn = get_db_connection()
    if not conn:
        return []  # Retorna lista vazia se não conectar

    conversas = []
    # Retorna resultados como dicionários
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT id, titulo FROM conversas ORDER BY data_criacao DESC")
        conversas = cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"Erro ao listar conversas: {err}")
    finally:
        cursor.close()
        conn.close()
    return conversas


def criar_nova_conversa(titulo="Nova Conversa"):
    """Cria uma nova conversa no banco e retorna seu ID."""
    conn = get_db_connection()
    if not conn:
        return None

    new_id = None
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO conversas (titulo) VALUES (%s)", (titulo,))
        conn.commit()
        new_id = cursor.lastrowid  # Pega o ID da conversa que acabou de ser criada
        print(f"Nova conversa criada com ID: {new_id}")
    except mysql.connector.Error as err:
        print(f"Erro ao criar nova conversa: {err}")
    finally:
        cursor.close()
        conn.close()
    return new_id


def carregar_mensagens(id_conversa):
    """Carrega as mensagens de uma conversa específica e retorna no formato do LangChain."""
    if id_conversa is None:
        return []  # Se não há conversa selecionada, retorna histórico vazio

    conn = get_db_connection()
    if not conn:
        return []  # Retorna lista vazia se não conseguir conectar

    mensagens_db = []
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT role, content
            FROM mensagens
            WHERE id_conversa = %s
            ORDER BY data_envio ASC
        """, (id_conversa,))
        mensagens_db = cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"Erro ao carregar mensagens da conversa {id_conversa}: {err}")
    finally:
        cursor.close()
        conn.close()

    # Converte do formato do banco ('role', 'content') para o formato do LangChain
    mensagens_langchain = []
    for msg in mensagens_db:
        if msg['role'] == 'human':
            mensagens_langchain.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'ai':
            mensagens_langchain.append(AIMessage(content=msg['content']))

    return mensagens_langchain


def salvar_mensagem(id_conversa, role, content):
    """Salva uma única mensagem no banco de dados."""
    # Adiciona uma verificação para não salvar mensagens vazias ou inválidas
    if not id_conversa or not content or not content.strip() or role not in ['human', 'ai']:
        print(
            f"Tentativa de salvar mensagem inválida ignorada (id: {id_conversa}, role: {role}, content: '{content[:20]}...')")
        return False

    conn = get_db_connection()
    if not conn:
        return False  # Indica que falhou

    success = False
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO mensagens (id_conversa, role, content)
            VALUES (%s, %s, %s)
        """, (id_conversa, role, content))
        conn.commit()
        success = True
        # Debug
        print(f"Mensagem salva (id_conversa: {id_conversa}, role: {role})")
    except mysql.connector.Error as err:
        print(f"Erro ao salvar mensagem na conversa {id_conversa}: {err}")
    finally:
        cursor.close()
        conn.close()
    return success

def deletar_conversa(id_conversa):
    """Deleta uma conversa e suas mensagens (usando ON DELETE CASCADE)."""
    conn = get_db_connection()
    if not conn:
        return False

    success = False
    cursor = conn.cursor()
    try:
        # Graças ao ON DELETE CASCADE na tabela mensagens,
        # apagar a conversa apagará as mensagens associadas.
        cursor.execute("DELETE FROM conversas WHERE id = %s", (id_conversa,))
        conn.commit()
        # Verifica se alguma linha foi realmente afetada (deletada)
        if cursor.rowcount > 0:
            success = True
            print(f"Conversa ID {id_conversa} deletada com sucesso.")
        else:
            print(f"Nenhuma conversa encontrada com ID {id_conversa} para deletar.")

    except mysql.connector.Error as err:
        print(f"Erro ao deletar conversa ID {id_conversa}: {err}")
    finally:
        cursor.close()
        conn.close()
    return success

# --- Bloco para Teste (Opcional) ---
# Se você rodar este arquivo diretamente (python db.py), ele cria as tabelas.
if __name__ == "__main__":
    print("Verificando/Criando tabelas do banco de dados...")
    criar_tabelas()
    print("\nTeste do db.py concluído.")
