import streamlit as st
import numpy as np
import cv2
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="AI Image Filter App", page_icon="🖼️", layout="centered")
st.title("AI Image Filter App")
st.markdown("Upload an image, get AI filter suggestions, then apply your chosen filter.")

# Session State to handle streamlit automatically refreshing whenever we click on a widget
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "ai_answer" not in st.session_state:
    st.session_state.ai_answer = ""
if "filtered_image" not in st.session_state:
    st.session_state.filtered_image = None

# Upload Image
uploaded_image = st.file_uploader("Upload your file (JPG, JPEG, PNG)", type=["jpg","jpeg","png"])
if uploaded_image:
    st.session_state.original_image = uploaded_image

if st.session_state.original_image:
    image = Image.open(st.session_state.original_image)
    st.image(image, caption="Original Image", use_container_width=True)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Compute image metrics
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness_score = np.mean(gray)
    contrast = gray.std()
    st.write(f"**Blur:** {blur_score:.2f}, **Brightness:** {brightness_score:.2f}, **Contrast:** {contrast:.2f}")

    # AI Suggestion 
    def get_ai_input(blur_score, brightness_score, contrast):
        if not OPENAI_API_KEY:
            return "⚠️ No API key detected. AI suggestions are disabled."
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            prompt = f"""
            The image has a blur score of {blur_score:.2f}, brightness {brightness_score:.2f}, and contrast {contrast:.2f}.
            Provide a single **best filter recommendation** from: Gaussian Blur, Median Blur, Low Pass, High Pass,
            and explain why. Be concise and confident.
            """
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role":"system","content":"You are an expert AI image filter advisor."},
                    {"role":"user","content":prompt},
                ],
                temperature=1,
                max_completion_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ AI suggestion failed: {str(e)}"

    if st.button("Get AI Filter Suggestion"):
        st.session_state.ai_answer = get_ai_input(blur_score, brightness_score, contrast)

    if st.session_state.ai_answer:
        st.markdown("### 🤖 AI Suggestion")
        st.markdown(st.session_state.ai_answer)

    # Filter Selection 
    st.markdown("---")
    st.markdown("### Choose Filter to Apply")
    filter_type = st.selectbox("Select a filter", ["Gaussian Blur", "Median Blur", "Low Pass", "High Pass"]).lower()
    ksize = st.slider("Kernel Size (odd numbers)", 1, 31, 5, step=2)

    # ---------------- Apply Filter ----------------
    def apply_filter(image_cv, filter_type, ksize):
        if filter_type == 'gaussian blur':
            return cv2.GaussianBlur(image_cv, (ksize, ksize), 0)
        elif filter_type == 'low pass':
            return cv2.blur(image_cv, (ksize, ksize))
        elif filter_type == 'median blur':
            return cv2.medianBlur(image_cv, ksize)
        elif filter_type == 'high pass':
            blur = cv2.GaussianBlur(image_cv, (ksize, ksize), 0)
            return cv2.addWeighted(image_cv, 1.5, blur, -0.5, 0)
        else:
            return image_cv

    if st.button("Apply Filter"):
        filtered = apply_filter(image_cv, filter_type, ksize)
        rgb_filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
        st.session_state.filtered_image = rgb_filtered

    # Showing Filtered Image
    if st.session_state.filtered_image is not None:
        st.markdown("### Filtered Image")
        st.image(st.session_state.filtered_image, use_container_width=True)
