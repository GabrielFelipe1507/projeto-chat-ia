import google.generativeai as genai
import os
from dotenv import load_dotenv

# Carrega a chave do .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Erro: Chave GEMINI_API_KEY não encontrada no arquivo .env")
else:
    try:
        print("Configurando a API do Google...")
        genai.configure(api_key=api_key)

        print("\nListando modelos disponíveis para sua chave:")
        # Pede ao Google a lista de modelos que suportam 'generateContent'
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")

        print("\nTeste concluído.")

    except Exception as e:
        print(f"\nOcorreu um erro ao tentar listar os modelos:")
        print(e)