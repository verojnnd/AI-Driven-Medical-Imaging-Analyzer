import os
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
import streamlit as st

# Set your API Key (Replace with your actual key)
GOOGLE_API_KEY = "AIzaSyDfu4HlCI_MFjPgee9WBH35k1qpVOmVjb4"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Ensure API Key is provided
if not GOOGLE_API_KEY:
    raise ValueError("⚠️ Please set your Google API Key in GOOGLE_API_KEY")

# Initialize the Medical Agent
medical_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGoTools()],
    markdown=True
)

# Medical Analysis Query
query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the medical image and structure your response as follows:

### 1. Image Type & Region
- Identify imaging modality (X-ray/MRI/CT/Ultrasound/etc.).
- Specify anatomical region and positioning.
- Evaluate image quality and technical adequacy.

### 2. Key Findings
- Highlight primary observations systematically.
- Identify potential abnormalities with detailed descriptions.
- Include measurements and densities where relevant.

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level.
- List differential diagnoses ranked by likelihood.
- Support each diagnosis with observed evidence.
- Highlight critical/urgent findings.

### 4. Patient-Friendly Explanation
- Simplify findings in clear, non-technical language.
- Avoid medical jargon or provide easy definitions.
- Include relatable visual analogies.

### 5. Research Context
- Use DuckDuckGo search to find recent medical literature.
- Search for standard treatment protocols.
- Provide 2-3 key references supporting the analysis.

Ensure a structured and medically accurate response using clear markdown formatting.
"""

def analyze_medical_image(image_path):
    """Processes and analyzes a medical image using AI."""
    image = PILImage.open(image_path)
    width, height = image.size
    aspect_ratio = width / height
    new_width = 500
    new_height = int(new_width / aspect_ratio)
    resized_image = image.resize((new_width, new_height))
    temp_path = "temp_resized_image.png"
    resized_image.save(temp_path)
    agno_image = AgnoImage(filepath=temp_path)
    try:
        response = medical_agent.run(query, images=[agno_image])
        return response.content
    except Exception as e:
        return f"⚠️ Analysis error: {e}"
    finally:
        os.remove(temp_path)

# Streamlit UI setup
st.set_page_config(page_title="Medical Imaging Analyzer", layout="centered")
st.title("🧬 AI-Driven Medical Imaging Analyzer")

# Inisialisasi riwayat chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat UI
st.markdown(
    """
    Upload a medical image (X-ray, MRI, CT, Ultrasound, etc.) and click **Analyze** to get an AI-powered diagnostic report.
    """
)

uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png", "bmp", "gif"], label_visibility="visible")

analyze_clicked = st.button("Analyze", use_container_width=True, disabled=uploaded_file is None)
import io

if analyze_clicked and uploaded_file is not None:
    # Simpan gambar ke BytesIO
    image_bytes = uploaded_file.getvalue()
    # Tampilkan gambar user
    st.session_state.chat_history.append({
        "role": "user",
        "content": "Uploaded an image for analysis.",
        "image_bytes": image_bytes
    })
    # Simpan sementara ke file untuk analisis
    image_path = f"temp_image.{uploaded_file.type.split('/')[1]}"
    with open(image_path, "wb") as f:
        f.write(image_bytes)
    # Analisis gambar
    with st.spinner("🔍 Analyzing the image... Please wait."):
        report = analyze_medical_image(image_path)
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": report
    })
    os.remove(image_path)

# Tampilkan riwayat chat (seperti ChatGPT)
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            if "image_bytes" in msg:
                st.image(io.BytesIO(msg["image_bytes"]), caption="Your uploaded image", use_container_width=True)
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"], unsafe_allow_html=True)
            
# Jika belum upload, tampilkan peringatan
if not st.session_state.chat_history and uploaded_file is None:
    st.info("Please upload a medical image and click Analyze to begin.")