import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.pyplot as plt
import io
from pathlib import Path
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import pandas as pd
import time

# Configuración rápida
st.set_page_config(page_title="LacunaSense", page_icon=":material/cardiology:", layout="centered")

# Configurar rutas con auto-resolve (para que funcione donde sea)
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "keras_model.h5"
LOGO_PATH = BASE_DIR / "assets" / "LacunaSense.png"

IMG_SIZE = (224, 224)
FS = 250  # sampling frequency for CSV preview

# Colores
BG_COLOR = "#f7f7f7"
CARD_COLOR = "#ffffff"
ACCENT = "#4d8aff"
TEXT_COLOR = "#333333"

# CSS inyectado
st.markdown(f"""
    <style>
    body {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
        font-family: 'Segoe UI', sans-serif;
    }}
    .card {{
        background-color: {CARD_COLOR};
        padding: 18px;
        border-radius: 10px;
        box-shadow: 0px 3px 8px rgba(0,0,0,0.06);
    }}
    .title {{
        font-weight:700;
        font-size:22px;
        text-align: center;
    }}
    .muted {{
        color:#6b6b6b;
    }}
    </style>
""", unsafe_allow_html=True)


# Carga del modelo
@st.cache_resource
def load_tm_model():
    return tf.keras.models.load_model(str(MODEL_PATH))

model = load_tm_model()


# Estados de sesión para comportamiento single page
if "screen" not in st.session_state:
    st.session_state.screen = "home"

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# Functiones de elementos gráficos
def predict_image_from_pil(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    label_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    label = "Señal apropiada" if label_idx == 0 else "Señal contaminada"
    return label, confidence, img


def create_pdf_report(result_dict, logo_path=LOGO_PATH):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4

    margin_x = 50
    current_y = h - 60

    # Logo
    try:
        logo_img = Image.open(logo_path)
        logo_reader = ImageReader(logo_img)
        logo_w = 120
        logo_h = 120 * (logo_img.height / logo_img.width)
        c.drawImage(logo_reader, margin_x, current_y - logo_h,
                    width=logo_w, height=logo_h, mask='auto')
    except:
        logo_w = 0

    # Titulos
    c.setFont("Helvetica-Bold", 18)
    title_x = margin_x + logo_w + 20
    c.drawString(title_x, current_y - 20, "LacunaSense — ECG Signal Grading Report")

    c.setFont("Helvetica", 10)
    c.drawString(title_x, current_y - 40,
                 f"Generado: {result_dict.get('timestamp', '')}")

    current_y = current_y - logo_h - 10
    c.line(margin_x, current_y, w - margin_x, current_y)

    current_y -= 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin_x, current_y, "Resumen del Resultado")

    current_y -= 22
    c.setFont("Helvetica", 12)
    label = result_dict.get("label", "N/A")
    conf = result_dict.get("confidence", 0.0)
    c.drawString(margin_x, current_y, f"Grado de la Señal: {label}")
    c.drawRightString(w - margin_x, current_y,
                      f"Confianza: {conf:.2f}")

    current_y -= 30
    c.setFont("Helvetica", 11)
    c.drawString(margin_x, current_y, "Notas:")
    current_y -= 18

    if label == "Señal apropiada":
        c.drawString(margin_x + 8, current_y,
                     "La señal no presenta interferencia por artefactos")
    else:
        c.drawString(margin_x + 8, current_y,
                     "La señal presenta ruido u artefactos, considere retomar la lectura o calibrar su equipo.")

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(margin_x, 50,
                 "LacunaSense — Prototype grading system. Not a diagnostic device.")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# Sidebar
st.sidebar.image(str(LOGO_PATH), width=120)
st.sidebar.markdown("<h2>LacunaSense</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

if st.sidebar.button("Inicio",type="secondary"):
    st.session_state.screen = "home"
if st.sidebar.button("Análisis",type="secondary"):
    st.session_state.screen = "upload"
if st.sidebar.button("Historial",type="secondary"):
    st.session_state.screen = "history"
if st.sidebar.button("Sobre de",type="secondary"):
    st.session_state.screen = "about"

st.sidebar.markdown("---")
st.sidebar.markdown("<small class='muted'>LacunaSense MVP &#169; 2025 </small>", unsafe_allow_html=True)

# Header de sesión
top_left, top_space, top_right = st.columns([1,6,2])

with top_right:
    if st.button("Invitado", icon=":material/person:", use_container_width=True):
        st.toast("Función no disponible: Contenido post-beta")
        


# Pantallas renderizables
def render_home():
    card = st.container(border=True, height=400, vertical_alignment="center", horizontal_alignment="center")
    card.image(str(LOGO_PATH), width=160)
    card.markdown("<h1 class='title'>LacunaSense</h1>", unsafe_allow_html=True)
    card.markdown("<div style='text-align: center;'><p>Systema de graduación de integridad de señales ECG Prototipo</p></div", unsafe_allow_html=True)
    # ---- Centered Horizontal Buttons ----
    col1, col2, col3, = card.columns([1,1,1])

    with col1:
        if st.button("Analizar", icon=":material/search:", width='stretch'):
            st.session_state.screen = "upload"
            st.rerun()

    with col2:
        if st.button("Historial de análisis", icon=":material/folder:", width='stretch'):
            st.session_state.screen = "history"
            st.rerun()

    with col3:
        if st.button("Sobre nosotros",icon=":material/info:", width='stretch'):
            st.session_state.screen = "about"
            st.rerun()

def render_upload():
    card =st.container(border=True)
    card.title(body="Medir Calidad de ECG")
    card.subheader("Conectar con dispositivo ECG")

    card.write(
        "Puedes conectar un dispositivo ECG compatible para transmitir la señal en tiempo real. "
        "Esta funcionalidad está en fase beta."
    )

    if card.button("Conectar dispositivo ECG", icon=":material/cable:"):
        st.toast("🔧 Función en beta: compatibilidad con dispositivos ECG próximamente.")

    card.markdown("<hr>", unsafe_allow_html=True)
    card.subheader("Medir señal capturada en formato de imagen")
    uploaded = card.file_uploader("Sube una imagen de longitud de onda o un CSV compatible", type=["png", "jpg", "jpeg", "csv"])

    if uploaded:
        ext = uploaded.name.split(".")[-1].lower()
        card.markdown("<hr>", unsafe_allow_html=True)

        if ext in ["png", "jpg", "jpeg"]:
            card.image(uploaded, width='stretch')
            pil_img = Image.open(uploaded)

            if card.button("Analizar"):
                with st.spinner("Analizando señal ECG..."):
                    time.sleep(2)
                    label, confidence, _ = predict_image_from_pil(pil_img)

                ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                result = {
                    "label": label,
                    "confidence": confidence,
                    "timestamp": ts,
                    "source": uploaded.name,
                    "image": pil_img 
                }

                st.session_state.last_result = result
                st.session_state.history.append(result)
                st.session_state.screen = "result"
                st.rerun()

        elif ext == "csv":
            ecg = np.loadtxt(uploaded, delimiter=",")
            t = np.linspace(0, len(ecg)/FS, len(ecg))

            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(t, ecg)
            card.pyplot(fig)

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            pil_img = Image.open(buf)

            if card.button("Grade ECG"):
                with st.spinner("Analyzing ECG signal..."):
                    label, confidence, _ = predict_image_from_pil(pil_img)

                ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                result = {
                    "label": label,
                    "confidence": confidence,
                    "timestamp": ts,
                    "source": uploaded.name,
                    "image": pil_img 
                }

                st.session_state.last_result = result
                st.session_state.history.append(result)
                st.session_state.screen = "result"


def render_result():
    card =st.container(border=True, horizontal_alignment="center")
    res = st.session_state.last_result

    if not res:
        card.warning("No result available.")
        return

    label = res["label"]
    conf = res["confidence"]
    img = res.get("image", None)

    color = "#2E7D32" if label == "Señal apropiada" else "#C62828"


    card.markdown(f"<h1>Calidad estimada: <h2 style='color:{color}'>{label}</h2></h1>", unsafe_allow_html=True)
    card.markdown(f"<p>Índice de confianza: {conf:.2f}</p>", unsafe_allow_html=True)
    card.markdown(f"<p class='muted'>{res['timestamp']}</p>", unsafe_allow_html=True)

    # renderizar imagen previa
    if img:
        card.markdown("<hr>", unsafe_allow_html=True)
        card.subheader("Señal analizada")
        card.image(img, width="stretch")
    
    pdf_bytes = create_pdf_report(res)
    name = f"LacunaSense_Report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    col1, col2 = st.columns([1,1])
    with col1:
        card.download_button("Descargar reporte en PDF", data=pdf_bytes, file_name=name, mime="application/pdf")
    with col2:
        card.button("Medir otra señal ECG", on_click=lambda: st.session_state.__setitem__("screen", "upload"))



# def render_history():
#     with st.container(border=True):
#         st.markdown("<h2>Historial</h2>", unsafe_allow_html=True)

#         if not st.session_state.history:
#             st.info("No existen registros previos en esta sesión.")
#         else:
#             for h in reversed(st.session_state.history):
#                 st.markdown(f"**{h['label']}** — Confianza: {h['confidence']:.2f} — {h['timestamp']}")
#                 st.markdown("<hr>", unsafe_allow_html=True)

def render_history():
    with st.container(border=True):
        st.markdown("<h2>Registro de análisis</h2>", unsafe_allow_html=True)

        history = st.session_state.history

        if not history:
            st.info("No existen registros previos en esta sesión.")
            return

        # Convert history into a DataFrame
        df = pd.DataFrame(history)

        # Rename columns for presentation
        df = df.rename(columns={
            "label": "Clasificación",
            "confidence": "Confianza",
            "timestamp": "Fecha/Hora",
            "source": "Origen"
        })

        # Optional: round confidence values
        df["Confianza"] = df["Confianza"].astype(float).round(2)

        # Reorder columns (optional)
        df = df[["Clasificación", "Confianza", "Fecha/Hora", "Origen"]]

        # Display table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


def render_about():
    with st.container(border=False, gap="small", height=300, vertical_alignment="center"):
        st.markdown("<h2>Sobre LacunaSense</h2>", unsafe_allow_html=True)
        st.write("LacunaSense es un sistema de medición de integridad de calidad de señal de tipo ECG optimizado a equipos biomédicos. Por el momento no es un dispositivo de diagnostico medico certificado.")



# ROUTER (Single-page)
if st.session_state.screen == "home":
    render_home()
elif st.session_state.screen == "upload":
    render_upload()
elif st.session_state.screen == "result":
    render_result()
elif st.session_state.screen == "history":
    render_history()
elif st.session_state.screen == "about":
    render_about()
else:
    render_home()
