# -*- coding: utf-8 -*-
"""
José People Analytics | Streamlit
Versión preparada para ejecutar como app.py.
Instalación: pip install -r requirements.txt
Ejecución: streamlit run app.py
"""
import io
import json
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="José People Analytics", page_icon="🧠", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
h1 { letter-spacing: -0.5px; }
section[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.2); }
div[data-testid="stMetric"] { border: 1px solid rgba(128,128,128,.2); padding: 12px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 José People Analytics")
st.caption("Asistente de análisis de talento humano con inteligencia artificial")
st.write("Analiza encuestas, comentarios, perfiles de candidatos y otros datos de Recursos Humanos.")

with st.sidebar:
    st.header("⚙️ Configuración")
    provider = st.selectbox("Proveedor de IA", ["OpenRouter (opción gratuita)", "OpenAI (requiere saldo/créditos)"])
    if provider.startswith("OpenRouter"):
        api_key = st.text_input("OpenRouter API Key", type="password", help="Pega tu clave personal de OpenRouter, normalmente empieza por sk-or-v1-.")
        model = st.selectbox("Modelo de OpenRouter", [
            "meta-llama/llama-3.3-70b-instruct:free",
            "deepseek/deepseek-r1:free",
            "google/gemma-3-27b-it:free",
        ], index=0, help="Los modelos gratuitos disponibles pueden cambiar; si uno no está disponible, prueba otro de la lista.")
        base_url = "https://openrouter.ai/api/v1"
    else:
        api_key = st.text_input("OpenAI API Key", type="password", help="Tu clave se utiliza solo para las solicitudes de esta sesión.")
        model = st.selectbox("Modelo de OpenAI", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"], index=0)
        base_url = None
    st.divider()
    uploaded = st.file_uploader("Carga tus datos (CSV o Excel)", type=["csv", "xlsx"])
    st.caption("Consejo: usa datos anonimizados y evita subir información personal sensible de colaboradores o candidatos.")

def get_client():
    if not api_key:
        if provider.startswith("OpenRouter"):
            st.warning("Introduce tu clave API de OpenRouter en el panel izquierdo para activar las funciones con IA.")
        else:
            st.warning("Introduce tu clave API de OpenAI en el panel izquierdo para activar las funciones con IA.")
        return None
    try:
        from openai import OpenAI
        if provider.startswith("OpenRouter"):
            return OpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers={"HTTP-Referer": "https://streamlit.io", "X-Title": "Jose People Analytics"},
            )
        return OpenAI(api_key=api_key)
    except ImportError:
        st.error("Falta instalar la librería openai. Ejecuta: pip install openai")
        return None

def ask_ai(prompt, system="Eres especialista en Recursos Humanos, People Analytics y análisis organizacional. Responde en español, con claridad y sin inventar datos."):
    client = get_client()
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        st.error(f"No se pudo completar la solicitud a {provider}: {exc}")
        return None

def download_csv(df, filename="resultado"):
    data = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("⬇️ Descargar resultados CSV", data=data, file_name=f"{filename}.csv", mime="text/csv")

df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
        if df.empty:
            st.error("El archivo no contiene filas.")
            st.stop()
        with st.sidebar:
            st.success(f"Archivo cargado: {len(df):,} filas · {len(df.columns)} columnas")
        with st.expander("👀 Vista previa de los datos"):
            st.dataframe(df.head(10), use_container_width=True)
    except Exception as exc:
        st.error(f"No se pudo leer el archivo. Verifica que sea un CSV o Excel válido. Detalle: {exc}")
        st.stop()
else:
    st.info("Para comenzar, carga un archivo CSV o Excel en el panel izquierdo. También puedes explorar el constructor de prompts sin cargar datos.")
    df = pd.DataFrame()

section = st.sidebar.selectbox("📌 Herramienta", [
    "Inicio",
    "Análisis de sentimiento",
    "Extracción de datos clave",
    "Riesgo de rotación",
    "Clusterización de texto",
    "Asistente de reclutamiento",
    "Detección de temas",
    "Constructor de prompts HR/NLP",
])

if section == "Inicio":
    st.subheader("Tu espacio de trabajo de People Analytics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Herramientas", "7")
    c2.metric("Formatos admitidos", "CSV · XLSX")
    c3.metric("IA generativa", "OpenRouter / OpenAI")
    st.markdown("""
    **¿Qué puedes hacer aquí?**
    - Analizar el sentimiento de comentarios y encuestas.
    - Extraer información estructurada de descripciones.
    - Identificar señales textuales asociadas al riesgo de rotación.
    - Agrupar textos por similitud.
    - Comparar perfiles con una descripción de puesto.
    - Resumir temas recurrentes y construir prompts profesionales.

    **Importante:** los resultados de IA son orientativos. No utilices puntuaciones automáticas como única base para contratar, despedir, promover o tomar decisiones que afecten a una persona.
    """)

elif section == "Constructor de prompts HR/NLP":
    st.subheader("🧰 Constructor de prompts profesionales HR/NLP")
    objetivo = st.selectbox("Objetivo", [
        "Encuesta de clima laboral", "Experiencia del colaborador", "Entrevista por competencias",
        "Desarrollo del talento", "Reclutamiento y selección", "Análisis de rotación",
        "Evaluación de capacitación", "Diversidad e inclusión", "Otro"
    ])
    contexto = st.text_area("Contexto de la organización o necesidad", placeholder="Describe brevemente el caso...")
    datos = st.text_area("Datos disponibles y limitaciones", placeholder="Ej.: respuestas anónimas de una encuesta; no hay datos demográficos...")
    formato = st.selectbox("Formato de salida", ["Informe ejecutivo", "Tabla estructurada", "Lista de recomendaciones", "Preguntas de análisis", "Plan de acción"])
    if st.button("✨ Construir prompt"):
        prompt = f"""Actúa como especialista en Recursos Humanos y People Analytics.
Objetivo: {objetivo}
Contexto: {contexto or 'No especificado'}
Datos disponibles y limitaciones: {datos or 'No especificados'}
Formato requerido: {formato}
Analiza solo la información proporcionada, diferencia hechos de hipótesis, señala limitaciones y propone recomendaciones prácticas, éticas y medibles. No inventes datos ni infieras atributos sensibles."""
        st.text_area("Prompt listo para copiar", value=prompt, height=260)
        st.download_button("Descargar prompt TXT", prompt, file_name="prompt_people_analytics.txt", mime="text/plain")

elif df.empty:
    st.warning("Carga primero un archivo CSV o Excel para utilizar esta herramienta.")
else:
    text_cols = list(df.select_dtypes(include=["object", "string"]).columns)
    all_cols = list(df.columns)

    if section == "Análisis de sentimiento":
        st.subheader("📊 Análisis de sentimiento")
        st.write("Clasifica comentarios como positivos, neutros o negativos.")
        if not text_cols:
            st.error("El archivo no contiene columnas de texto.")
        else:
            col = st.selectbox("Columna de comentarios", text_cols)
            limit = st.number_input("Máximo de filas a procesar", min_value=1, max_value=len(df), value=min(len(df), 50))
            if st.button("▶ Analizar sentimiento"):
                results = []
                progress = st.progress(0)
                for i, value in enumerate(df[col].head(int(limit)).fillna("")):
                    if not str(value).strip():
                        results.append("Sin texto")
                    else:
                        answer = ask_ai(f'Clasifica el sentimiento del comentario en Positivo, Neutro o Negativo. Devuelve solo la categoría.\nComentario: {str(value)[:4000]}')
                        results.append(answer.strip() if answer else "Error")
                    progress.progress((i + 1) / int(limit))
                out = df.head(int(limit)).copy()
                out["sentimiento_ia"] = results
                st.bar_chart(out["sentimiento_ia"].value_counts())
                st.dataframe(out, use_container_width=True)
                download_csv(out, "analisis_sentimiento")

    elif section == "Extracción de datos clave":
        st.subheader("📌 Extracción de datos clave")
        if not text_cols:
            st.error("El archivo no contiene columnas de texto.")
        else:
            col = st.selectbox("Columna con descripciones", text_cols)
            fields = st.text_input("Campos que deseas extraer, separados por comas", "formación, habilidades, experiencia")
            limit = st.number_input("Máximo de filas a procesar", min_value=1, max_value=len(df), value=min(len(df), 50), key="extract_limit")
            if st.button("▶ Extraer información"):
                field_list = [x.strip() for x in fields.split(",") if x.strip()]
                if not field_list:
                    st.warning("Escribe al menos un campo.")
                else:
                    output = []
                    bar = st.progress(0)
                    for i, value in enumerate(df[col].head(int(limit)).fillna("")):
                        if not str(value).strip():
                            output.append({f: "" for f in field_list})
                        else:
                            ans = ask_ai("Extrae los campos solicitados de este texto. Devuelve exclusivamente JSON válido, con una clave por campo. Si falta información, usa 'No especificado'.\nCampos: " + json.dumps(field_list, ensure_ascii=False) + "\nTexto:\n" + str(value)[:5000])
                            try:
                                parsed = json.loads(ans or "{}")
                                output.append({f: parsed.get(f, "No especificado") for f in field_list})
                            except Exception:
                                output.append({f: (ans or "No disponible") if j == 0 else "" for j, f in enumerate(field_list)})
                        bar.progress((i + 1) / int(limit))
                    out = pd.concat([df.head(int(limit)).reset_index(drop=True), pd.DataFrame(output)], axis=1)
                    st.dataframe(out, use_container_width=True)
                    download_csv(out, "extraccion_datos")

    elif section == "Riesgo de rotación":
        st.subheader("⚠️ Análisis textual de señales relacionadas con rotación")
        st.warning("Es un análisis exploratorio de comentarios, no una predicción fiable de la conducta futura de una persona. No debe utilizarse para decisiones laborales individuales.")
        if not text_cols:
            st.error("El archivo no contiene columnas de texto.")
        else:
            col = st.selectbox("Columna de comentarios", text_cols)
            limit = st.number_input("Máximo de filas a procesar", min_value=1, max_value=len(df), value=min(len(df), 50), key="risk_limit")
            if st.button("▶ Analizar comentarios"):
                categories, reasons = [], []
                bar = st.progress(0)
                for i, value in enumerate(df[col].head(int(limit)).fillna("")):
                    ans = ask_ai(f"""Analiza el texto de forma prudente y solo como señal agregada de experiencia laboral.
Etiqueta la expresión observada como: Señal explícita de insatisfacción, Señal ambigua, Sin señal explícita o Sin texto.
Da una justificación descriptiva breve. No infieras intención real de renunciar ni rasgos personales.
Formato: ETIQUETA | JUSTIFICACIÓN
Texto: {str(value)[:4000]}""") if str(value).strip() else "Sin texto | No hay comentario"
                    parts = (ans or "No disponible | No se pudo analizar").split("|", 1)
                    categories.append(parts[0].strip())
                    reasons.append(parts[1].strip() if len(parts) > 1 else "")
                    bar.progress((i + 1) / int(limit))
                out = df.head(int(limit)).copy()
                out["senal_textual"] = categories
                out["justificacion"] = reasons
                st.bar_chart(out["senal_textual"].value_counts())
                st.dataframe(out, use_container_width=True)
                download_csv(out, "senales_rotacion")

    elif section == "Clusterización de texto":
        st.subheader("🔍 Clusterización exploratoria de texto")
        if not text_cols:
            st.error("El archivo no contiene columnas de texto.")
        else:
            col = st.selectbox("Columna de texto", text_cols)
            k = st.slider("Número de grupos", min_value=2, max_value=8, value=3)
            if st.button("▶ Crear grupos"):
                texts = df[col].fillna("").astype(str)
                if len(df) < k or texts.str.strip().eq("").all():
                    st.error("Se necesitan suficientes filas con texto para crear los grupos.")
                else:
                    try:
                        vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
                        matrix = vectorizer.fit_transform(texts)
                        model_k = KMeans(n_clusters=k, random_state=42, n_init=10)
                        out = df.copy()
                        out["grupo_textual"] = model_k.fit_predict(matrix)
                        st.bar_chart(out["grupo_textual"].value_counts().sort_index())
                        st.dataframe(out, use_container_width=True)
                        download_csv(out, "clusters_texto")
                    except Exception as exc:
                        st.error(f"No se pudieron crear los grupos: {exc}")

    elif section == "Asistente de reclutamiento":
        st.subheader("🎯 Asistente de reclutamiento")
        st.warning("Utiliza la herramienta como apoyo para estructurar una revisión humana. No automatices rechazos ni tomes decisiones solo con el resultado de IA.")
        jd = st.text_area("Descripción del puesto (responsabilidades y requisitos)", height=180)
        cv_col = st.selectbox("Columna que contiene el CV o perfil", text_cols or all_cols)
        name_col = st.selectbox("Columna de nombre o identificador", ["(ninguna)"] + all_cols)
        limit = st.number_input("Máximo de perfiles a revisar", min_value=1, max_value=len(df), value=min(len(df), 30), key="recruit_limit")
        if st.button("▶ Comparar perfiles"):
            if not jd.strip():
                st.warning("Introduce la descripción del puesto.")
            else:
                rows = []
                bar = st.progress(0)
                for i, (_, row) in enumerate(df.head(int(limit)).iterrows()):
                    name = str(row[name_col]) if name_col != "(ninguna)" else f"Perfil {i+1}"
                    ans = ask_ai(f"""Compara el perfil con los requisitos del puesto usando solo evidencia explícita. No infieras edad, género, etnia, salud, discapacidad, religión u otros atributos sensibles. No decidas contratar ni descartar.
Puesto:\n{jd[:5000]}\nPerfil:\n{str(row[cv_col])[:5000]}
Devuelve JSON válido con: coincidencias (lista), brechas (lista), preguntas_para_validar (lista), ajuste_estimado (0 a 100, orientativo), explicacion (máximo 60 palabras). Si no hay evidencia, indícalo.""")
                    try:
                        data = json.loads(ans or "{}")
                    except Exception:
                        data = {"explicacion": ans or "No disponible"}
                    rows.append({"identificador": name, **{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in data.items()}})
                    bar.progress((i + 1) / int(limit))
                out = pd.DataFrame(rows)
                st.dataframe(out, use_container_width=True)
                download_csv(out, "revision_perfiles")

    elif section == "Detección de temas":
        st.subheader("🧩 Detección de temas recurrentes")
        if not text_cols:
            st.error("El archivo no contiene columnas de texto.")
        else:
            col = st.selectbox("Columna de respuestas o comentarios", text_cols)
            limit = st.number_input("Máximo de textos a incluir", min_value=1, max_value=len(df), value=min(len(df), 100), key="topics_limit")
            if st.button("▶ Detectar temas"):
                sample = df[col].head(int(limit)).dropna().astype(str).tolist()
                if not sample:
                    st.warning("No hay textos para analizar.")
                else:
                    prompt = "Analiza estos comentarios e identifica entre 3 y 6 temas recurrentes. Para cada tema incluye nombre, descripción breve, ejemplos representativos y una recomendación práctica. Distingue patrones observados de hipótesis y no inventes porcentajes.\n\n" + "\n".join(f"{i+1}. {t[:1500]}" for i, t in enumerate(sample))
                    ans = ask_ai(prompt)
                    if ans:
                        st.markdown(ans)
                        st.download_button("Descargar informe TXT", ans, file_name="deteccion_temas.txt", mime="text/plain")