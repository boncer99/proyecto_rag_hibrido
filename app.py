#!pip install pymupdf langchain langchain-community langchain-elasticsearch langchain-openai
#!pip install python-dotenv
#!pip install langgraph

# LO nuevo para el poryecto final
#!pip install llama-index 
#!pip install -qU llama-index-vector-stores-elasticsearch  o  !pip install -qU llama-index-vector-stores-elasticsearch

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore

#from langchain_core.prompts import ChatPromptTemplate ORIGINAL HASTA ANTES DE ESTE EJEMPLO
from langchain.prompts import ChatPromptTemplate   # este en su reemplazo porque funcionaba con esta libreria

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
#from langchain_community.document_loaders import PyMuPDFLoader
#from langchain.text_splitter import CharacterTextSplitter


import sys
from flask import Flask, jsonify, request
from langchain.tools import tool

from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from flask import Flask, request, render_template, jsonify

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit


# llama index
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.elasticsearch import ElasticsearchStore as le  #uso variable le
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
# fin de llama index

import os
from dotenv import load_dotenv
load_dotenv()

# ========= CONFIGURACIÓN ==========

# Obtener las variables del entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")    # para llm = ChatOpenAI(openai_api_key=openai_api_key)

ES_URL = os.getenv("ES_URL")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
INDEX_NAME = os.getenv("INDEX_NAME")

# Establecer la clave de OpenAI como variable de entorno
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY    # no estaba originalmente


# ===================== FLASK ======================
def obtener_genero(pelicula_name):
    prompt = ChatPromptTemplate.from_template("Indica en una palabra de qué género es la película: {pelicula}")
    #llm = ChatOpenAI()
    #llm = ChatOpenAI(openai_api_key=openai_api_key)
    llm = ChatOpenAI(model="gpt-4-0125-preview", openai_api_key=openai_api_key,verbose=True)
    parser = StrOutputParser()
    pipeline = RunnableSequence(prompt | llm | parser)
    return pipeline.invoke({"pelicula": pelicula_name})

app = Flask(__name__)



# ======================Herramienta de recuperacion de documentos ================================#
@tool
def tool_general(query: str) -> str:
    """  Usa esta herramienta para responder preguntas sobre los documentos disponibles: 
        Detalles Técnicos del sistema, Asistente inteligente para uso del sistema y  Plan de escalabilidad'. 
        Los documentos están en español y tratan sobre domótica, uso del asistente y mejora del sistema. """
    # ===================== CONEXIÓN A VECTOR STORE =====================
    vector_store = le(
    es_url=ES_URL,
    es_user=ES_USER,
    es_password=ES_PASSWORD,
    index_name= INDEX_NAME,
    )

    storage_context_read = StorageContext.from_defaults(vector_store=vector_store)

    index_read = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context_read,embed_model=OpenAIEmbedding()
    )

    retriever = index_read.as_retriever(search_kwargs={"k": 4}) # devuelve los 4 fragmentos mas relevantes
    results = retriever.retrieve(query)

    output = "\n\n".join([
        f"[{i+1}] {node.node.text.strip()}\nFuente: {node.node.metadata.get('source', 'N/A')}"
        for i, node in enumerate(results)
    ])
    return output


# ========= AGENTE DE RESPUESTA ==========
#modelo = ChatOpenAI(model="gpt-4-0125-preview")
#modelo = ChatOpenAI(model="gpt-4-0125-preview", openai_api_key=openai_api_key)
modelo = ChatOpenAI(model="gpt-4-0125-preview", openai_api_key=openai_api_key,verbose=True)

system_prompt = """
Eres un asistente técnico. Solo responde usando la información disponible en los documentos.
No inventes respuestas. Si no encuentras información relevante, di que no se encontró en los documentos.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{messages}"),
    ]
)

# Crea el agente con la tool general
toolkit = [tool_general]
agent = create_react_agent(model=modelo, tools=toolkit, prompt=prompt)

# =========  FUNCION CONSULTAR (EJEMPLO) ==========
from langchain_core.messages import HumanMessage

import uuid
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
#config = {"configurable": {"thread_id": "consulta_001"}}

def ejecutar_consulta(pregunta: str) -> str:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    # Inicializar la respuesta como string vacío
    respuesta = ""
    # Ejecutar la consulta con el agente en modo streaming
    for step in agent.stream(
        {"messages": [HumanMessage(content=pregunta)]},
        config,
        stream_mode="values",
    ):
        # Obtener el contenido del último mensaje generado
        mensaje = step["messages"][-1].content
        # Acumular en la respuesta
        respuesta = mensaje
    return respuesta



### funcion flask para consulta usando las herramientas generadas ############33
@app.route('/info', methods=['GET', 'POST'])
def consulta():
    resultado_solicitud =""
    if request.method == 'POST':
        texto_consulta = request.form.get('solicitud')
        resultado_solicitud = ejecutar_consulta(texto_consulta)
    return render_template('info.html', resultado_solicitud=resultado_solicitud)

### API Expuesta para consultas usando las herramientas generadas ############33
# uso es http://localhost:5000/consulta_docs?q=¿Qué incluye el plan de escalabilidad?
@app.route('/consulta_docs', methods=['GET'])
def consulta_rapida():
    pregunta = request.args.get('q')
    if not pregunta:
        return jsonify({"error": "Falta el parámetro 'q' con la consulta"}), 400
    respuesta = ejecutar_consulta(pregunta)
    return jsonify({"respuesta": respuesta})



#necesario si se ejecuta de manera local , en consola python app.py
#if __name__ == "__main__":
#    app.run(debug=True)