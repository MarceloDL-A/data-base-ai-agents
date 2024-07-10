import json
import os
from dotenv import load_dotenv  # Biblioteca para carregar variáveis de ambiente a partir de um arquivo .env
from langchain_openai import ChatOpenAI  # Biblioteca para interação com os modelos de linguagem da OpenAI
import pandas as pd  # Biblioteca para manipulação de dados
from sqlalchemy import create_engine  # Biblioteca para interação com bancos de dados SQL
from openai import OpenAI  # Cliente oficial da OpenAI para chamadas à API
import streamlit as st  # Biblioteca para criação de aplicativos web interativos
import helpers  # Módulo customizado com funções auxiliares
from helpers import (
    get_avg_salary_and_female_count_for_division,
    get_total_overtime_pay_for_department,
    get_total_longevity_pay_for_grade,
    get_employee_count_by_gender_in_department,
    get_employees_with_overtime_above,
)
from langdetect import detect  # Biblioteca para detectar o idioma da pergunta

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Recupera a chave da API do OpenAI a partir das variáveis de ambiente
openai_key = os.getenv("OPENAI_API_KEY")

# Define o modelo de linguagem que será utilizado
llm_name = "gpt-3.5-turbo"
model = ChatOpenAI(api_key=openai_key, model=llm_name)

# Cria um cliente para chamadas à API do OpenAI
client = OpenAI(api_key=openai_key)

# Importações específicas para o uso com bases de dados SQL
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

# Define o caminho para o arquivo de banco de dados SQLite
database_file_path = "./db/salary.db"

# Cria um motor de conexão ao banco de dados SQLite
engine = create_engine(f"sqlite:///{database_file_path}")
file_url = "./data/salaries_2023.csv"
os.makedirs(os.path.dirname(database_file_path), exist_ok=True)

# Carrega os dados do arquivo CSV e preenche valores nulos com 0
df = pd.read_csv(file_url).fillna(value=0)
# Transfere os dados do DataFrame para a tabela no banco de dados SQL
df.to_sql("salaries_2023", con=engine, if_exists="replace", index=False)

# Definição dos prompts prefixo e sufixo
SQL_PROMPT_PREFIX = """
First set the pandas display options to show all the columns,
get the column names, then answer the question.
"""

SQL_PROMPT_SUFFIX = """
- **ALWAYS** before giving the Final Answer, try another method.
Then reflect on the answers of the two methods you did and ask yourself
if it answers correctly the original question.
If you are not sure, try another method.
FORMAT 4 FIGURES OR MORE WITH COMMAS.
- If the methods tried do not give the same result, reflect and
try again until you have two methods that have the same result.
- If you still cannot arrive to a consistent result, say that
you are not sure of the answer.
- If you are sure of the correct answer, create a beautiful
and thorough response using Markdown.
- **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
- **ALWAYS**, as part of your "Final Answer", explain how you got
to the answer on a section that starts with: "\n\nExplanation:\n".
In the explanation, mention the column names that you used to get
to the final answer.
- Provide the final answer in the same language as the question.
"""

SQL_PROMPT_PREFIX_PT = """
Primeiro, defina as opções de exibição do pandas para mostrar todas as colunas,
obtenha os nomes das colunas e, em seguida, responda à pergunta.
"""

SQL_PROMPT_SUFFIX_PT = """
- **SEMPRE** antes de dar a Resposta Final, tente outro método.
Reflita sobre as respostas dos dois métodos que você fez e pergunte a si mesmo
se responde corretamente à pergunta original.
Se você não tiver certeza, tente outro método.
FORMATE 4 DÍGITOS OU MAIS COM VÍRGULAS.
- Se os métodos tentados não derem o mesmo resultado, reflita e
tente novamente até ter dois métodos que tenham o mesmo resultado.
- Se ainda assim não conseguir chegar a um resultado consistente, diga que
não tem certeza da resposta.
- Se você tiver certeza da resposta correta, crie uma resposta bonita
e completa usando Markdown.
- **NÃO INVENTE UMA RESPOSTA OU USE CONHECIMENTO PRÉVIO,
USE APENAS OS RESULTADOS DOS CÁLCULOS QUE VOCÊ FEZ**.
- **SEMPRE**, como parte de sua "Resposta Final", explique como você chegou
à resposta em uma seção que começa com: "\n\nExplicação:\n".
Na explicação, mencione os nomes das colunas que você usou para chegar
à resposta final.
- Forneça a resposta final no mesmo idioma da pergunta.
"""

# Função que gerencia a conversa com o modelo
def run_conversation(query, language):
    if language == 'pt':
        prefix = SQL_PROMPT_PREFIX_PT
        suffix = SQL_PROMPT_SUFFIX_PT
    else:
        prefix = SQL_PROMPT_PREFIX
        suffix = SQL_PROMPT_SUFFIX

    # Cria uma lista de mensagens com o prefixo, a consulta e o sufixo
    messages = [{"role": "user", "content": prefix + query + suffix}]

    # Chama o modelo com a conversa e as funções disponíveis
    response = client.chat.completions.create(
        model=llm_name,
        messages=messages,
        tools=helpers.tools_sql,
        tool_choice="auto",
    )
    response_message = response.choices[0].message

    # Verifica se há chamadas de ferramentas no retorno do modelo
    tool_calls = response_message.tool_calls
    if tool_calls:
        # Dicionário de funções disponíveis
        available_functions = {
            "get_avg_salary_and_female_count_for_division": get_avg_salary_and_female_count_for_division,
            "get_total_overtime_pay_for_department": get_total_overtime_pay_for_department,
            "get_total_longevity_pay_for_grade": get_total_longevity_pay_for_grade,
            "get_employee_count_by_gender_in_department": get_employee_count_by_gender_in_department,
            "get_employees_with_overtime_above": get_employees_with_overtime_above,
        }
        messages.append(response_message)

        # Para cada chamada de ferramenta, executa a função correspondente
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            if function_name == "get_employees_with_overtime_above":
                function_response = function_to_call(amount=function_args.get("amount"))
            elif function_name == "get_total_longevity_pay_for_grade":
                function_response = function_to_call(grade=function_args.get("grade"))
            else:
                function_response = function_to_call(**function_args)
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": str(function_response),
            })
            # Obtém uma nova resposta do modelo após a execução das funções
            second_response = client.chat.completions.create(
                model=llm_name,
                messages=messages,
            )
        return second_response

# Interface Streamlit
st.title("Database AI Agent with LangChain")

# Exibe uma prévia do dataset
st.write("### Dataset Preview")
st.write(df.head())

# Detecta o idioma da pergunta para garantir a formatação correta
question = st.text_input(
    "Enter your question about the dataset:",
    "What is the total longevity pay for employees with the grade 'M3'?",
    key='initial_question'
)
language = detect(question)

# Ajusta a interface de acordo com o idioma detectado
if language == 'pt':
    st.write("### Pergunte Algo")
    button_label = "Executar Consulta"
    final_answer_label = "### Resposta Final"
else:
    st.write("### Ask a Question")
    button_label = "Run Query"
    final_answer_label = "### Final Answer"

# Executa a consulta e exibe o resultado
if st.button(button_label):
    res = run_conversation(question, language)
    
    st.write(final_answer_label)
    st.markdown(res.choices[0].message.content)