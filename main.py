import gradio as gr
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model("deepfake_model.keras")
IMG_SIZE = (256, 256)

def predict_image(image):
    img = np.array(image)
    img = tf.image.resize(img, IMG_SIZE) / 255.0
    img = tf.expand_dims(img, axis=0)
    pred = model.predict(img, verbose=0)[0][0]
    
    # Updated Threshold
    threshold = 0.7
    label = "REAL" if pred > threshold else "FAKE"
    confidence = pred if label == "REAL" else 1 - pred

    return label, f"{confidence*100:.2f}%"

# Custom CSS
css = """
.gradio-container {
    background-color: #121212;
    color: #f5f5f5;
    font-family: 'Segoe UI', sans-serif;
}
.header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(90deg, #2c3e50 0%, #4ca1af 100%);
    color: white;
    border-radius: 8px;
    margin-bottom: 20px;
}
.section-card {
    background: #1e1e1e;
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 18px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}
table {
    width: 100%;
    border-collapse: collapse;
}
table th, table td {
    border: 1px solid #888;
    padding: 8px;
    text-align: left;
}
ul li {
    margin-bottom: 6px;
}
"""

# HTML content
project_title = """
<div class="section-card">
<h2>🛡️ Project Title: DeepGuard AI – Deepfake Detection System</h2>
</div>
"""

project_purpose = """
<div class="section-card">
<h3>🎯 Project Purpose</h3>
<p>DeepGuard AI is developed to identify whether a facial image is real or synthetically generated (deepfake) using deep learning. The rise of AI-generated content has raised ethical, legal, and societal concerns. This project aims to:</p>
<ul>
<li>Detect tampered or manipulated facial images.</li>
<li>Help users, media platforms, and law enforcement distinguish fake identities.</li>
<li>Fight misinformation, identity theft, and digital fraud caused by deepfakes.</li>
</ul>
</div>
"""

key_features = """
<div class="section-card">
<h3>💡 Key Features</h3>
<ul>
<li>✅ Real vs. Fake Classification – Upload an image, and the system predicts whether it's genuine or AI-generated.</li>
<li>📊 Confidence Score – Shows how confident the model is in its prediction (as a percentage).</li>
<li>🧠 CNN-based Model – Uses a convolutional neural network trained on real vs. fake facial images.</li>
<li>🌐 Intuitive Web Interface – Built with Gradio for ease of access and usage — no coding needed for users.</li>
<li>⚡ Fast Inference – Lightweight and optimized to run predictions in real-time.</li>
<li>🌐 Web-Based Interface – Intuitive and clean UI built with Gradio — no need for technical skills to operate.</li>
</ul>
</div>
"""

tech_stack = """
<div class="section-card">
<h3>🧠 Technology Stack</h3>
<table>
<tr><th>Component</th><th>Technology</th></tr>
<tr><td>Frontend/UI</td><td>Gradio</td></tr>
<tr><td>Backend</td><td>Python</td></tr>
<tr><td>Machine Learning</td><td>TensorFlow, Keras</td></tr>
<tr><td>Image Preprocessing</td><td>NumPy, PIL, tf.image</td></tr>
<tr><td>Model Type</td><td>CNN (Convolutional Neural Network)</td></tr>
<tr><td>Deployment</td><td>Local / Web via Gradio</td></tr>
</table>
</div>
"""

applications = """
<div class="section-card">
<h3>🚀 Applications</h3>
<ul>
<li><strong>Media & Journalism:</strong> Verifies authenticity of visual content before publishing.</li>
<li><strong>Social Media Platforms:</strong> Flags fake user profile images or AI-generated avatars.</li>
<li><strong>Law Enforcement:</strong> Assists in identifying fake IDs or digitally altered evidence.</li>
<li><strong>Corporate Identity Verification:</strong> Validates users during KYC onboarding.</li>
<li><strong>Public Awareness & Education:</strong> Helps people recognize dangers of deepfake technology.</li>
</ul>
</div>
"""

short_model_explanation = """
<div class="section-card">
<h3>🧠 Model Explanation</h3>
<p>DeepGuard AI uses a CNN model trained on real and AI-generated faces. It analyzes subtle facial cues to spot deepfakes, focusing on:</p>
<ul>
<li><strong>Micro-expressions:</strong> Tiny facial movements often missed by fakes.</li>
<li><strong>Lighting:</strong> Checks for unnatural shadows/highlights.</li>
<li><strong>Texture:</strong> Detects skin or hair distortions.</li>
<li><strong>Biological cues:</strong> Identifies missing signs like blood flow color shifts.</li>
</ul>
<p>These features pass through convolution layers to classify the image as <strong>REAL</strong> or <strong>FAKE</strong> with a confidence score.</p>
</div>
"""

# Full App Layout
with gr.Blocks(css=css, title="DeepGuard AI") as app:
    with gr.Tabs():
        with gr.Tab("🧾 Project Overview"):
            gr.Markdown("""<div class="header"><h1>DeepGuard AI</h1><p>Deepfake Detection – Project Overview</p></div>""")
            gr.Markdown(project_title)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(project_purpose)
                    gr.Markdown(key_features)
                with gr.Column(scale=1):
                    gr.Markdown(tech_stack)
                    gr.Markdown(applications)
                    

            # Add Project Flowchart Section
            with gr.Column():
                gr.Markdown("""<div class="section-card"><h3>📊 Project Flowchart</h3>
                <p>This diagram illustrates the flow of DeepGuard AI from Input to Output.</p></div>""")
                gr.Image(value="flowchart_of_project.png", label="DeepGuard AI - Workflow Diagram", show_label=True, width=800)

        with gr.Tab("🔍 Deepfake Detector"):
            gr.Markdown("""<div class="header"><h1>DeepGuard AI</h1><p>Upload an image to check if it's real or fake</p></div>""")
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="Upload Image", type="pil")
                    submit_btn = gr.Button("Analyze", variant="primary")
                    gr.Markdown("""
                    <div class="section-card">
                    <h3>📸 Tips for Better Accuracy</h3>
                    <ul>
                    <li>Use clear, front-facing photos</li>
                    <li>Prefer high-resolution images</li>
                    <li>Try images with good lighting</li>
                    </ul>
                    </div>
                    """)
                with gr.Column():
                    label_output = gr.Label(label="Prediction")
                    confidence_output = gr.Label(label="Confidence Score")
                    gr.Markdown(short_model_explanation)

            submit_btn.click(
                fn=predict_image,
                inputs=image_input,
                outputs=[label_output, confidence_output]
            )

if __name__ == "__main__":
    app.launch()
