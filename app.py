import streamlit as st
from PIL import Image, ImageDraw
import os
import time
from src.predictor import setup_models, predict

if "models" not in st.session_state:
    pipe, seg_image_processor, image_segmentor, mlsd_processor = setup_models()
    st.session_state.models = {
        "pipe": pipe,
        "seg_image_processor": seg_image_processor,
        "image_segmentor": image_segmentor,
        "mlsd_processor": mlsd_processor,
    }
else:
    models = st.session_state.models
    pipe = models["pipe"]
    seg_image_processor = models["seg_image_processor"]
    image_segmentor = models["image_segmentor"]
    mlsd_processor = models["mlsd_processor"]

st.set_page_config(
    page_title="Int.AI",
    page_icon=":house:",
    initial_sidebar_state="collapsed",
    layout="wide",
)
# custom css
css = """
<style>
    .block-container{
        padding-top: 2rem;
        padding-bottom: 1rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-size:0.8em;
    }
    div.stButton {
        height:50%;
        width: 50%;
        font-size: 10px;
        paddin: 0.5em 1em;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# Example prompts
alpha_prompt = "create modern designed light blue themed living room"
alpha_negative = "No red colors, no dark colors, no clutter"
beta_prompt = "make the luxury bedroom look more elegant and tastefully decorated with a queen size bed against the wall and light bluish green theme"
beta_negative = "None"
gamma_prompt = "elegant bedroom with a TV facing the bed and a sofa by the window"
gamma_negative = ""
sigma_prompt = "luxurious living room with a green sofa"
sigma_negative = ""

# Some states and variables
st.session_state.image_ready = False
st.session_state.submit_pressed = False
st.session_state.example_showing = False
st.session_state.uploaded_file = None


# Directory for storing uploaded images and their outputs
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_uploaded_file(uploaded_file):
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def show_example(example: int):
    reset()
    st.session_state.example_showing = True
    example_prompts = [alpha_prompt, beta_prompt, gamma_prompt, sigma_prompt]
    example_negatives = [alpha_negative, beta_negative, gamma_negative, sigma_negative]

    st.markdown(
        f":red[Prompt:] {example_prompts[example]}",
        unsafe_allow_html=True,
    )
    st.markdown(
        f":red[Negative Prompt:] {example_negatives[example]}",
        unsafe_allow_html=True,
    )
    col_upload, col_processed = st.columns([1, 1], gap="small")
    with col_upload:
        st.markdown(
            "<h4 style='text-align: center;'>Uploaded Image</h4>",
            unsafe_allow_html=True,
        )
        st.image(f"tests/{example}.jpg", use_container_width=True)
    with col_processed:
        st.markdown(
            "<h4 style='text-align: center;'>Processed Image</h4>",
            unsafe_allow_html=True,
        )
        st.image(f"tests/{example}_out.png", use_container_width=True)


# Simulated model processing function
def process_image_with_prompt(image_path, prompt):
    time.sleep(2)  # Simulate processing time
    output_path = os.path.join(OUTPUT_DIR, f"output_{os.path.basename(image_path)}")
    image = Image.open(image_path)
    processed_image = image.copy()

    # Example: Add text overlay (simulate processing with prompt)
    processed_image = processed_image.convert("RGBA")
    overlay = Image.new("RGBA", processed_image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    draw.text((10, 10), prompt, fill=(255, 0, 0, 255))  # Red text for the prompt
    processed_image = Image.alpha_composite(processed_image, overlay)

    processed_image = processed_image.convert("RGB")  # Remove alpha for saving
    processed_image.save(output_path)
    return output_path


def reset():
    st.session_state.uploaded_file = None
    st.session_state.image_ready = False
    st.session_state.submit_pressed = False
    st.session_state.example_showing = False


# Main App
col_side, col_main = st.columns([2, 5], gap="medium")
with col_side:
    st.markdown(
        "<h3 style='text-align: left;'>&#127968<u>Interior AI</u> </h3>",
        unsafe_allow_html=True,
    )
    st.write("Upload an image, input a prompt and get your desired design.")

    # Input section
    st.session_state.uploaded_file = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg"]
    )
    if st.session_state.uploaded_file:
        st.session_state.image_ready = True

    prompt = st.text_input("Enter a prompt to guide the processing")

    # advanced options
    with st.expander("Advanced Options"):
        neg_prompt = st.text_input("Negative Prompts", value="")

    submit_button = st.button("Submit")
    if submit_button:
        if prompt and st.session_state.image_ready:
            st.session_state.submit_pressed = True
        elif not prompt and not st.session_state.image_ready:
            st.write(":red[Select file and enter prompt OR select an example]")
        elif not prompt:
            st.write(":red[Enter the prompt]")
        else:
            st.write(":red[Invalid Request]")


with col_main:
    if st.session_state.image_ready:
        col_upload, col_processed = st.columns([1, 1], gap="small")
        with col_upload:
            st.markdown(
                "<h4 style='text-align: center;'>Uploaded Image</h4>",
                unsafe_allow_html=True,
            )
            if st.session_state.uploaded_file:
                # Save uploaded image
                img_path = os.path.join(UPLOAD_DIR, st.session_state.uploaded_file.name)
                with open(img_path, "wb") as f:
                    f.write(st.session_state.uploaded_file.getbuffer())
                st.image(img_path, use_container_width=True)

        with col_processed:
            # Processing section
            st.markdown(
                "<h4 style='text-align: center;'>Processed Image</h4>",
                unsafe_allow_html=True,
            )
            if st.session_state.submit_pressed:
                with st.spinner("Generating... :red[Will take around 2 minutes]"):
                    output_path = predict(
                        pipe,
                        image_path=img_path,
                        prompt=prompt,
                        negative_prompt=neg_prompt if neg_prompt else "",
                        seg_image_processor=seg_image_processor,
                        image_segmentor=image_segmentor,
                        mlsd_processor=mlsd_processor,
                        seed=None,
                        output_dir="outputs",
                    )

                st.image(output_path, use_container_width=True)

    st.markdown("### Usage Examples ", unsafe_allow_html=True)
    st.markdown(
        "<p style = 'text-align : left; font-size: 1em; color: red;'>* make sure there is no data uploaded before selecting an example.</p>",
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4 = st.columns(4, gap="small")

    if col1.button(
        "Example Alpha", key="ex1", help="Use Example 0", use_container_width=True
    ):
        show_example(0)
    if col2.button(
        "Example Beta", key="ex2", help="Use Example 1", use_container_width=True
    ):
        show_example(1)
    if col3.button(
        "Example Gamma", key="ex3", help="Use Example 2", use_container_width=True
    ):
        show_example(2)
    if col4.button(
        "Example Sigma", key="ex4", help="Use Example 3", use_container_width=True
    ):
        show_example(3)
