# Sistema de asistencia basado en Agentes Inteligentes y RAG para interacción natural integrado en sistema domótico 

## 🧠 Objetivo

Desarrollar un sistema de asistencia domótica basado en modelos de lenguaje generativo (LLM), accesible y contextual, que permita a personas con movilidad reducida interactuar de forma natural con dispositivos del hogar a través de consultas en lenguaje natural.

---

## ❗ Problemática a Resolver

Las personas con movilidad reducida enfrentan desafíos para interactuar con dispositivos electrónicos del hogar. Los sistemas domóticos tradicionales no contemplan completamente interfaces accesibles ni naturales. Esto limita la autonomía del usuario y requiere soluciones de asistencia que combinen accesibilidad, control y monitoreo inteligente.

Además, existe una barrera cognitiva y operativa relacionada con la comprensión y uso de estos sistemas. Muchos carecen de interfaces intuitivas que permitan conocer funciones disponibles, ejecución de comandos o estado de dispositivos.

Para abordar esta necesidad, el presente proyecto integra un modelo de lenguaje generativo (LLM), como OpenAI, que actúa como un asistente conversacional accesible y contextual. Este agente interpreta consultas como “¿Qué gestos puedo usar?” o “¿Cómo apago la luz?” y responde de manera personalizada, conectada al estado actual del sistema, facilitando así el aprendizaje, la interacción y la autonomía.

---

## 🧱 Arquitectura

La arquitectura general del sistema permite consultas en lenguaje natural sobre el estado y funcionamiento de un sistema domótico inteligente. El flujo es el siguiente:

1. **Interfaz Web**: Desarrollada en Flask y desplegada en Railway. Permite enviar consultas vía formulario (`/info`) o peticiones GET (`/consulta_docs`).
2. **Agente Conversacional**: Basado en LangChain y un LLM que interpreta la intención del usuario. Sigue un enfoque RAG (Retrieval-Augmented Generation).
3. **Herramientas del Agente**:
   - **LlmaIndex + Elasticsearch**: Para acceder a documentos técnicos embebidos como vectores semánticos.
   - **PostgreSQL**: Para recuperar datos en tiempo real como temperatura, dispositivos activos, historial, etc.
   - **Control de Dispositivos**: Normaliza la consulta, detecta intención y ajusta configuración de dispositivos (luces, ventanas, etc.).
4. **Flask**: Encapsula toda la lógica del sistema y gestiona las rutas de acceso y agentes.
5. **Variables de entorno**: Se almacenan de forma segura en un archivo `.env`.


![1arquitectura](https://github.com/user-attachments/assets/b7bb0c2a-afaf-4aa7-a503-7093a15e514e)
<p align="center">Figura 1. Diagrama de la arquitectura</p>

También se desarrolló un programa en Python que prepara documentos PDF para su almacenamiento en Elastic Cloud. Este script convierte los contenidos en vectores semánticos para búsquedas por similitud (RAG).

![2alamacena](https://github.com/user-attachments/assets/19490dc6-1794-4f89-b9de-d216cf12fc85)
<p align="center">Figura 2. Diagrama de la arquitectura para almacenamiento</p>

---

## ⚙️ Proceso de Instalación

### 1. Configurar Elasticsearch en Elastic Cloud

- Crear cuenta en [https://cloud.elastic.co](https://cloud.elastic.co)
- Crear una instancia de Elasticsearch.
- Obtener:
  - `ES_HOST`
  - `ES_USER`
  - `ES_PASSWORD`

### 2. Cargar documentos técnicos

- Usar el script de carga de PDFs con **LlamaIndex**.

### 3. Crear base de datos PostgreSQL en Railway

- Crear cuenta en [https://railway.app](https://railway.app)
- Crear un nuevo proyecto y añadir plugin **PostgreSQL**
- Copiar credenciales: `host`, `database`, `user`, `password`, `port`

### 4. (Opcional) Administrar base de datos con pgAdmin

- Crear nueva conexión con credenciales Railway.
- Crear o importar tablas (`sensores`, `historial_dispositivos`, etc.).

### 5. Subir proyecto a GitHub

Incluir:

- `main.py`
- `requirements.txt`
- `.env` (excluido en el repo público)
- scripts del agente

### 6. Desplegar en Railway

- Crear nuevo proyecto y seleccionar **"Deploy from GitHub repo"**.
- Configurar las variables de entorno desde la pestaña *Variables*.

### 7. Acceder desde Railway

- Usar el link público (Domain) generado por Railway.

---

## 🚀 Instrucciones de Despliegue

1. Configurar Elasticsearch en [Elastic Cloud](https://cloud.elastic.co)
2. Ejecutar el script Python para cargar los documentos técnicos al vector store.
3. Crear base de datos PostgreSQL en [Railway](https://railway.app)
4. Crear las tablas necesarias con pgAdmin o scripts SQL.
5. Subir proyecto a GitHub.
6. Desplegar proyecto desde Railway.
7. Configurar las variables de entorno.
8. Acceder a la aplicación desde el link público proporcionado por Railway.

---

## ✅ Cómo Probar

Una vez desplegado, se puede probar el sistema accediendo al dominio público generado por Railway. A través de la interfaz web, se ingresan consultas como:

- "¿Qué dispositivos están activos?"
- "¿Cómo se usa el gesto de encendido?"
- "¿Qué temperatura hay en la sala?"

El agente procesará la consulta, determinará la fuente de datos y devolverá una respuesta contextualizada.

![3result1](https://github.com/user-attachments/assets/fc3d2578-4def-4493-aaa7-c85c3e111a3e)
![3result2](https://github.com/user-attachments/assets/0578a18a-bebb-46ed-9a60-9f28471e9937)


---

## 🛠 Tecnologías Utilizadas

- Python
- Flask
- LangChain
- LlamaIndex
- OpenAI API
- PostgreSQL
- Elasticsearch (Elastic Cloud)
- Railway

---

## 🔒 Notas de Seguridad

Recuerda mantener el archivo `.env` fuera del repositorio público. Este archivo debe incluir credenciales y claves como:

```env
ES_HOST=
ES_USER=
ES_PASSWORD=
POSTGRES_HOST=
POSTGRES_DB=
POSTGRES_USER=
POSTGRES_PASSWORD=
OPENAI_API_KEY=
