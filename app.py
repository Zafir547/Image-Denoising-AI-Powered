import io
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras

# Page configuration
st.set_page_config(
    page_title="Image Denoising App",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        font-size: 16px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🖼️ Image Denoising AI-Powered")
st.markdown("Upload a noisy image and get a clean reconstructed.")

# Sidebar for settings
st.sidebar.header("⚙️ Settings")

# Model loading
@st.cache_resource
def load_model():
    # Load the trained U-Net model
    try:
        model = keras.models.load_model(
            'best_denoising_unet.keras',
            custom_objects={'combine_loss': lambda y_true, y_pred: 
                          0.7 * tf.keras.losses.MSE(y_true, y_pred) + 
                          0.3 * tf.keras.losses.MAE(y_true, y_pred)}
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure 'best_denoising_unet.keras' is in the same directory.")
        return None

# Image preprocessing functions
def preprocess_image(image, target_size=(128, 128)):
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize to [0, 1]
    img = img.astype('float32') / 255.0
    
    # Add channel dimension
    img = np.expand_dims(img, axis=-1)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def add_noise_to_image(image, noise_level=0.15):
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_img = image + noise
    noisy_img = np.clip(noisy_img, 0.0, 1.0)
    return noisy_img

def postprocess_image(image, enhance=True):
    # Remove batch and channel dimensions
    img = image[0].squeeze()
    
    # Convert to 8-bit
    img = (img * 255).astype(np.uint8)
    
    if enhance:
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        # Slight sharpening
        kernel = np.array([[-0.5,-0.5,-0.5],
                          [-0.5, 5,-0.5],
                          [-0.5,-0.5,-0.5]])
        img = cv2.filter2D(img, -1, kernel)
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    return img

# Load model
model = load_model()

if model is not None:
    st.sidebar.success("✅ Model loaded successfully!")
    
    # Noise settings
    add_noise = st.sidebar.checkbox("Add Artificial Noise", value=False)
    
    if add_noise:
        noise_level = st.sidebar.slider(
            "Noise Level",
            min_value=0.05,
            max_value=0.5,
            value=0.08,  # Lower default for better results
            step=0.01,
            help="Higher values add more noise. Recommended: 0.05-0.15"
        )
    
    # Enhancement settings
    st.sidebar.markdown("---")
    enhance_output = st.sidebar.checkbox(
        "Enhance Output", 
        value=True,
        help="Apply sharpening and contrast enhancement"
    )
    
    # File uploader
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload a grayscale or color image"
    )
    
    # Main content
    if uploaded_file is not None:
        # Read and display original image
        original_image = Image.open(uploaded_file)
        
        # Create columns for layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📥 Original Image")
            st.image(original_image, width='stretch')
        
        # Process button
        if st.button("🚀 Denoise Image", type="primary"):
            with st.spinner("Processing image..."):
                # Preprocess
                processed_img = preprocess_image(original_image)
                
                # Add noise if requested
                if add_noise:
                    noisy_img = add_noise_to_image(processed_img, noise_level)
                else:
                    noisy_img = processed_img
                
                # Run inference
                reconstructed = model.predict(noisy_img, verbose=0)
                
                # Postprocess
                noisy_display = postprocess_image(noisy_img, enhance=False)
                reconstructed_display = postprocess_image(reconstructed, enhance=enhance_output)
                
                # Display results
                with col2:
                    st.subheader("🔊 Noisy Image" if add_noise else "📥 Input Image")
                    st.image(noisy_display, width='stretch', clamp=True)
                
                with col3:
                    st.subheader("✨ Denoised Image")
                    st.image(reconstructed_display, width='stretch', clamp=True)
                
                # Quality metrics
                st.markdown("---")
                st.subheader("📊 Quality Metrics")
                
                # Calculate PSNR and SSIM if noise was added
                if add_noise:
                    from skimage.metrics import peak_signal_noise_ratio as psnr
                    from skimage.metrics import structural_similarity as ssim
                    
                    # Normalize original image properly
                    original_gray = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY)
                    original_gray = cv2.resize(original_gray, (128, 128))
                    original_gray = original_gray.astype('float32') / 255.0
                    
                    # Normalize comparison images
                    noisy_normalized = noisy_display.astype('float32') / 255.0
                    reconstructed_normalized = reconstructed_display.astype('float32') / 255.0
                    
                    # Calculate metrics with proper data_range
                    psnr_noisy = psnr(original_gray, noisy_normalized, data_range=1.0)
                    psnr_reconstructed = psnr(original_gray, reconstructed_normalized, data_range=1.0)
                    
                    ssim_noisy = ssim(original_gray, noisy_normalized, data_range=1.0)
                    ssim_reconstructed = ssim(original_gray, reconstructed_normalized, data_range=1.0)
                    
                    metric_col1, metric_col2 = st.columns(2)
                    
                    with metric_col1:
                        st.metric("PSNR Improvement", 
                                f"{psnr_reconstructed:.2f} dB",
                                f"+{psnr_reconstructed - psnr_noisy:.2f} dB")
                    
                    with metric_col2:
                        st.metric("SSIM Improvement",
                                f"{ssim_reconstructed:.4f}",
                                f"+{ssim_reconstructed - ssim_noisy:.4f}")
                
                # Download buttons
                st.markdown("---")
                st.subheader("💾 Download Results")
                
                download_col1, download_col2 = st.columns(2)
                
                with download_col1:
                    # Noisy image download
                    noisy_pil = Image.fromarray(noisy_display)
                    buf_noisy = io.BytesIO()
                    noisy_pil.save(buf_noisy, format='PNG')
                    btn_noisy = st.download_button(
                        label="📥 Download Noisy Image",
                        data=buf_noisy.getvalue(),
                        file_name="noisy_image.png",
                        mime="image/png"
                    )
                
                with download_col2:
                    # Reconstructed image download
                    reconstructed_pil = Image.fromarray(reconstructed_display)
                    buf_reconstructed = io.BytesIO()
                    reconstructed_pil.save(buf_reconstructed, format='PNG')
                    btn_reconstructed = st.download_button(
                        label="✨ Download Denoised Image",
                        data=buf_reconstructed.getvalue(),
                        file_name="denoised_image.png",
                        mime="image/png"
                    )
                
                st.success("✅ Processing complete!")
    
    else:
        # Instructions when no image is uploaded
        st.info("👈 Please upload an image using the sidebar to get started.")
        
        # Show example
        st.markdown("---")
        st.subheader("📖 How to Use")
        st.markdown("""
        1. **Upload an image** using the file uploader in the sidebar
        2. **(Optional)** Enable "Add Artificial Noise" to test denoising
        3. **Adjust noise level** if artificial noise is enabled
        4. **Click "Denoise Image"** to process
        5. **Compare results** and download the denoised image
        
        **Supported formats:** PNG, JPG, JPEG, BMP
        
        **Note:** The model works best with face images similar to the training data.
        """)

else:
    st.error("❌ Failed to load model. Please ensure 'best_denoising_unet.keras' is in the current directory.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with ❤️ using Streamlit and TensorFlow</p>
        <p>U-Net Denoising Model | SSIM: 79.89% | PSNR: 26.44 dB</p>
    </div>
    """, unsafe_allow_html=True)