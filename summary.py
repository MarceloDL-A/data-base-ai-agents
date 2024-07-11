import json
import os
from dotenv import load_dotenv  # Biblioteca para carregar variáveis de ambiente a partir de um arquivo .env
from langchain_openai import ChatOpenAI  # Biblioteca para interação com os modelos de linguagem da OpenAI
import pandas as pd  # Biblioteca para manipulação de dados
from sqlalchemy import create_engine  # Biblioteca para interação com bancos de dados SQL
from openai import OpenAI  # Cliente oficial da OpenAI para chamadas à API

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Recupera a chave da API do OpenAI a partir das variáveis de ambiente
openai_key = os.getenv("OPENAI_API_KEY")

# Define o modelo de linguagem que será utilizado
llm_name = "gpt-3.5-turbo"
model = ChatOpenAI(api_key=openai_key, model=llm_name)

# Cria um cliente para chamadas à API do OpenAI
client = OpenAI(api_key=openai_key)

file_url = "./data/salaries_2023.csv"

# Carrega os dados do arquivo CSV e preenche valores nulos com 0
df = pd.read_csv(file_url).fillna(value=0)


# Função para gerar o resumo do dataset
def generate_dataset_summary(df):
    column_summary = df.columns.tolist()
    description_summary = df.describe()
    null_counts_summary = df.isnull().sum()
    
    summary = f"""
    **Column Names:** {column_summary}
    
    **Statistics Overview:**\n
    {description_summary}
    
    **Null Values:**\n
    {null_counts_summary}
    """
    return summary

# Função para gerar um resumo de alto nível com o GPT
def generate_high_level_summary(summary = ""):

    
    prompt = f"""
    Based on the following dataset summary, generate a concise and professional high-level overview suitable for business professionals and data scientists:

    {summary}

    Don't use phrases with obvious information for this area. Only mention additional information if it is truly relevant.
    """
    messages = [{"role": "user", "content": prompt}]
    # Chama o modelo com a conversa e as funções disponíveis
    response = client.chat.completions.create(
        model=llm_name,
        messages=messages,
        temperature = 0.5

        
    )
    return response.choices[0].message.content


if __name__ == "__main__":
     summary = generate_dataset_summary(df)
     print(f"{generate_high_level_summary(summary = "")}")