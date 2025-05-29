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
uribd = os.getenv("DB_ACCESS")

ES_URL = os.getenv("ES_URL")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
INDEX_NAME = os.getenv("INDEX_NAME")

# Establecer la clave de OpenAI como variable de entorno
#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY    # no estaba originalmente SACAR PARA RAILWAY


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


# =============== conexion con la base de datos ==========================#
db_data = SQLDatabase.from_uri(uribd)
# Herramienta BD
toolkit_bd = SQLDatabaseToolkit(db=db_data,llm=ChatOpenAI(model="gpt-4-0125-preview", openai_api_key=openai_api_key,verbose=True))
tools_bd = toolkit_bd.get_tools()

# ================= herramientas de seleccion de tablas =================================#
def seleccionar_tabla_v2(question: str) -> str:
    """
    Selecciona la tabla correcta basada en la pregunta utilizando la LLM para inferir el nombre de la tabla,
    y luego verifica si el nombre de la tabla existe en las tablas disponibles.
    """
    # Obtiene las tablas disponibles desde la base de datos
    tablas_disponibles = db_data.get_table_names()  # Listado dinámico de tablas

    # Crea un mensaje que pasa la lista de tablas disponibles al modelo de lenguaje
    tablas_str = ", ".join(tablas_disponibles)  # Las tablas disponibles como un string

    # Consultamos a la LLM para obtener el nombre de la tabla basado en la pregunta y las tablas disponibles
    prompt = (f"Las siguientes tablas están disponibles en la base de datos: {tablas_str}. "
              f"Segun las tablas indicadas  ¿Devuelve sólo el nombre de la tabla a  la que corresponde esa pregunta: '{question}'?")
    #model = ChatOpenAI(openai_api_key=openai_api_key,verbose=True)
    model = ChatOpenAI(model="gpt-4-0125-preview", openai_api_key=openai_api_key,verbose=True)

    # Suponiendo que 'model' es el objeto que invoca el modelo de lenguaje
    #nombre_tabla_sugerido = model(prompt).strip().lower()  # Respuesta procesada a minúsculas
    respuesta = model.invoke([HumanMessage(content=prompt)])
    nombre_tabla_sugerido = respuesta.content.strip().lower()

    # Validamos si la tabla sugerida por el modelo existe en la base de datos
    #if nombre_tabla_sugerido not in tablas_disponibles:
    #    return f"[ERROR] La tabla '{nombre_tabla_sugerido}' no se reconoce. Las tablas disponibles son: {', '.join(tablas_disponibles)}"
    return nombre_tabla_sugerido
# Función para seleccionar la tabla correcta basada en la pregunta
def seleccionar_tabla(question: str) -> str:
    # Define las tablas disponibles
    tablas_disponibles = ['temperatura_registros', 'humedad_registros', 'calidad_aire_registros', 'sensores', 'historial_dispositivos', 'mqtt_topicos']
    # Aquí puedes agregar lógica de selección, por ejemplo, buscar palabras clave en la pregunta
    if "topico" in question.lower():                                                                ## proyecto final  2025, se cambio el orden
        return "mqtt_topicos"
    elif "topicos" in question.lower():
        return "mqtt_topicos"
    elif "tópicos" in question.lower():
        return "mqtt_topicos"
    elif "tópico" in question.lower():
        return "mqtt_topicos"
    if "temperatura" in question.lower():
        return "temperatura_registros"
    elif "humedad" in question.lower():
        return "humedad_registros"
    elif "calidad de aire" in question.lower():
        return "calidad_aire_registros"
    elif "sensor" in question.lower():
        return "sensores"
    elif "historial" in question.lower():
        return "historial_dispositivos"
    else:
        # Si no se encuentra una coincidencia clara, puede retornar una tabla predeterminada o lanzar un error
        return "temperatura_registros"  # Tabla predeterminada


# ====================================== herramientas para bases de datos con LangChain =======================================================#
from pydantic import BaseModel, Field
from langchain_core.tools import Tool
db_data = SQLDatabase.from_uri(uribd)
model = ChatOpenAI(model="gpt-4-0125-preview", openai_api_key=openai_api_key,verbose=True)
@tool
def get_schema(question: str) -> str:
    """Herramienta para recuperar el esquema de una tabla específica,si la tabla no existe devuelve un mensaje de error controlado
       Se deben usar otras herramientas SQL para usar el esquema y hacer la consulta SQL
    """
    # Usa la función 'seleccionar_tabla' para obtener el nombre de la tabla basándose en la pregunta
    tabla_seleccionada = seleccionar_tabla_v2(question)
    # Obtiene las tablas disponibles desde la base de datos
    tablas_disponibles = db_data.get_table_names()
    # Si la tabla seleccionada no existe en la base de datos, devuelve un mensaje de error controlado
    if tabla_seleccionada not in tablas_disponibles:
        return f"[ERROR] La tabla '{tabla_seleccionada}' no se reconoce. Las tablas disponibles son: {', '.join(tablas_disponibles)}"
    # Si la tabla es válida, devuelve el esquema de la tabla seleccionada
    schema = db_data.get_table_info([tabla_seleccionada])
    return schema

promptsql = ChatPromptTemplate.from_template("""
Basandonos en el esquema de tabla siguiente, escribe una consulta SQL (sin usar acentos ni corregir la ortografia) que responda a la pregunta del usuario:
    Tabla: {table}

    Esquema: {schema}

    Pregunta: {question}
    Sql Query:
""")

sqlchain = (
    promptsql
    | model.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

@tool
def generar_sql(question: str) -> str:
    """Genera una consulta SQL a partir del esquema, la pregunta y  usando la tabla adecuada seleccionada dinámicamente."""
    # Selecciona la tabla basándose en la pregunta
    tabla_seleccionada = seleccionar_tabla(question)
    schema = db_data.get_table_info([tabla_seleccionada])  # Solo obtiene el esquema de la tabla seleccionada
    return sqlchain.invoke({"schema": schema, "question": question,"table": tabla_seleccionada})

@tool
def run_query(query) -> str:
    """Herramienta que ejecuta una consulta SQL en la base de datos. Siempre se debe ejecutar el query sql"""
    resultado = db_data.run(query)
    if not resultado or resultado == [(None,)]:
        return "[SIN RESULTADOS] No se encontraron registros que coincidan con la consulta."
    return resultado

promptsqlquery = ChatPromptTemplate.from_template(
    """
    Basandonos en el esquema de tabla inferior, pregunta, SQL Query y Respuesta, escribe una respuesta en lenguaje natural:
    Tabla: {table}

    Esquema: {schema}

    Pregunta: {question}
    Sql Query: {sql_query}
    SQL Respuesta: {response}

    """)

sqlnatural_chain = (
     promptsqlquery
    | model
    | StrOutputParser()
)

@tool(return_direct=True)
def generar_respuesta(question: str, sql_query: str, response: str) -> str:
    """Genera una respuesta en lenguaje natural tomando el esquema, la consulta, la query,la respuesta de la ejecucion de la query y la tabla seleccionada
    “Cuando respondas a preguntas sobre tópicos MQTT, responde únicamente con el tópico, sin ningún texto adicional ni explicación.”
    """
    tabla_seleccionada = seleccionar_tabla(question)
    schema = db_data.get_table_info([tabla_seleccionada])
    return sqlnatural_chain.invoke({"schema": schema, "question": question, "sql_query":sql_query, "response":response, "table": tabla_seleccionada  })

# ==================================Fin de herramientas de  acceso  base de dato =====================================#



# ====================== Herramienta de recuperacion de documentos con LLamaIndex ================================#
@tool
def tool_general(query: str) -> str:                                                                 ## actualizado, proyecto final 2025
    """ Esta herramienta no sirve para consultas en las bases de datos SQL. Usa esta herramienta para responder preguntas sobre los documentos disponibles: 
        Detalles Técnicos del sistema, Asistente inteligente para uso del sistema y  Plan de escalabilidad'. 
        Los documentos están en español y tratan sobre domótica, uso del asistente y mejora del sistema.
         Ademas esta herramienta no da informacion de topicos mqtt de monitoreo, apagado o prendido.
         No uses acentos ni corrigas la ortografia en el texto generado"""
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



# ======================================== AGENTE DE RESPUESTA ======================================#
#modelo = ChatOpenAI(model="gpt-4-0125-preview")
#modelo = ChatOpenAI(model="gpt-4-0125-preview", openai_api_key=openai_api_key)
modelo = ChatOpenAI(model="gpt-4-0125-preview", openai_api_key=openai_api_key) #verbose=True)

system_prompt = """
Eres un asistente de bases de datos y de información técnica del sistema domótico en documentos. 
En este contexto la palabra tópico se refiere a la column topico que estan en la tabla mqtt_topicos. 
Si recibes consultas de información sobre las bases de datos, tablas o topicos utiliza las herramientas de consulta a base de datos para responder las preguntas sin usar acentos. 
Si recibes consultas de información de documentos o descripcion del funcionamiento del sistema solo responde usando la información disponible en los documentos.
No inventes respuestas. Si no encuentras información relevante, di que no se encontró.
No uses acentos ni corrigas la ortografia en el texto generado
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{messages}"),
    ]
)

# Crea el agente con la tool general
#toolkit = [tool_general, get_schema,generar_sql,run_query,generar_respuesta]
toolkit = [tool_general,generar_sql,run_query,generar_respuesta]
agent = create_react_agent(model=modelo, tools=toolkit, prompt=prompt)

#=============  Funciones para la deteccion de accion y  complementar solo la respuesta mqqt  =================#
def detectar_accion_llm(texto):
    prompt = f"""Extrae la intención principal del siguiente comando de usuario. 
Responde únicamente con "1" si se desea activar, abrir o encender algo, 
responde únicamente con "0" si se desea apagar, cerrar o desactivar algo.
Responde únicamente con None si no encuentra ninguna intencion anterior.
No uses acentos ni corrigas la ortografia en el texto generado.
Texto: "{texto}"
Respuesta:"""
    
    resultado = modelo.invoke([HumanMessage(content=prompt)])
    respuesta = resultado.content.strip()  # Accede al contenido de la respuesta

    #print(respuesta)
    if respuesta in ["0", "1"]:
        return respuesta
    return None

def detectar_accion(texto):
    texto = texto.lower()
    #print(texto)
    acciones_encender = ["enciende", "encender", "prende", "prender", "activa", "activar", "actives", "enciendas", "prendas"]
    acciones_apagar = ["apaga", "apagar", "desactiva", "desactivar", "apagues", "desactives"]
    
    for palabra in acciones_encender:
        if palabra in texto:
            #print("es 1")
            return "1"
    for palabra in acciones_apagar:
        if palabra in texto:
            #print("es 0")
            return "0"
    return None  # No se reconoce la acción

def generar_consulta_topico_llm(texto, valor):
    prompt = f"""Identifica la intencion del comando de usuario y sólo retira esas palabras de la oracion. 
    Pueden estar relacionadas a activar o encender algo, o apagar o desactivar algo, o abrir o cerrar algo
    En lugar de las palabras relacionadas a la intencion agrega palabras como "Devuelve el topico de" con el resto del texto de manera coherente
    No uses acentos ni corrigas la ortografia en el texto generado
Texto: "{texto}"
Respuesta:"""
    
    resultado = modelo.invoke([HumanMessage(content=prompt)])
    respuesta = resultado.content.strip()  # Accede al contenido de la respuesta

    return respuesta

import unidecode
#valor = ""
def normalizar_texto(texto):
    global valor
    texto = texto.lower()
    texto = unidecode.unidecode(texto)
    
    valor = detectar_accion_llm(texto) # si valor es diferente de None existe una intencion, si es asi debes eliminar esa intencion para obtener el topico
    
    if valor != None: # existe una intencion
        texto = generar_consulta_topico_llm(texto, valor) # si es 0 u 1 , debo reemplazar el texto de comando o intencion y agregar las palabras Devuelve el topico del elemento descrito
    
    return texto
#============= FIN de Funciones para la deteccion de accion y  complementar solo la respuesta mqqt  =================#


# =====================================  FUNCION CONSULTAR (EJEMPLO) =====================================#
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
        {"messages": [HumanMessage(content=normalizar_texto(pregunta))]},
        config,
        stream_mode="values",
    ):
        # Obtener el contenido del último mensaje generado
        mensaje = step["messages"][-1].content
        # Acumular en la respuesta
        respuesta = mensaje

    # si la devolucion es un topico concateno
    if valor != None:
        #topico_orden = step["messages"][-1].content + "/" + valor
        respuesta = respuesta + "/" + valor
        #print(topico_orden)
    #else:
        #step["messages"][-1].pretty_print()
    return respuesta



###================================= funcion flask para consulta usando las herramientas generadas ================##
@app.route('/info', methods=['GET', 'POST'])
def consulta():
    resultado_solicitud =""
    if request.method == 'POST':
        texto_consulta = request.form.get('solicitud')
        resultado_solicitud = ejecutar_consulta(texto_consulta)
    return render_template('info.html', resultado_solicitud=resultado_solicitud)

### API Expuesta para consultas usando las herramientas generadas ############
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