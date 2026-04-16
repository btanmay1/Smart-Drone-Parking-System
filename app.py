import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import os
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

# --- Page Config ---
st.set_page_config(page_title="Smart Drone Parking", page_icon="🚁", layout="wide")

# --- Load Model Make sure the notebook is run at least once to save it ---
MODEL_PATH = "smart_drone_parking_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model_artifact = load_model()

# --- Feature Extraction (from notebook) ---
IMG_SIZE = (64, 64)

def extract_features(image: np.ndarray) -> dict:
    if image is None:
        return None
    img_rgb = cv2.resize(image, IMG_SIZE)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    feats = {}
    for i, ch in enumerate(['R', 'G', 'B']):
        channel = img_rgb[:, :, i]
        feats[f'color_mean_{ch}']  = channel.mean()
        feats[f'color_std_{ch}']   = channel.std()
        feats[f'color_skew_{ch}']  = float(pd.Series(channel.flatten()).skew())
        
    feats['brightness']      = gray.mean()
    feats['contrast']        = gray.std()
    feats['saturation_mean'] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[:,:,1].mean()
    
    hog_feats = hog(gray, orientations=8, pixels_per_cell=(8,8),
                    cells_per_block=(2,2), feature_vector=True)
    feats['hog_mean'] = hog_feats.mean()
    feats['hog_std']  = hog_feats.std()
    feats['hog_max']  = hog_feats.max()
    for j, v in enumerate(hog_feats[:20]):
        feats[f'hog_{j}'] = v
        
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=10, density=True)
    for j, v in enumerate(lbp_hist):
        feats[f'lbp_{j}'] = v
        
    glcm = graycomatrix((gray // 16).astype(np.uint8), distances=[1],
                        angles=[0], levels=16, symmetric=True, normed=True)
    feats['glcm_contrast']     = graycoprops(glcm, 'contrast')[0,0]
    feats['glcm_dissimilarity']= graycoprops(glcm, 'dissimilarity')[0,0]
    feats['glcm_homogeneity']  = graycoprops(glcm, 'homogeneity')[0,0]
    feats['glcm_energy']       = graycoprops(glcm, 'energy')[0,0]
    feats['glcm_correlation']  = graycoprops(glcm, 'correlation')[0,0]
        
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    feats['edge_density'] = edges.mean() / 255.0
    feats['laplacian_var']= cv2.Laplacian(gray, cv2.CV_64F).var()
    
    eps = 1e-6
    feats['RG_ratio'] = feats['color_mean_R'] / (feats['color_mean_G'] + eps)
    feats['RB_ratio'] = feats['color_mean_R'] / (feats['color_mean_B'] + eps)
    feats['GB_ratio'] = feats['color_mean_G'] / (feats['color_mean_B'] + eps)
    feats['color_uniformity'] = 1 / (np.mean([feats['color_std_R'], feats['color_std_G'], feats['color_std_B']]) + eps)
    feats['bright_contrast'] = feats['brightness'] * feats['contrast']
    feats['texture_score'] = feats['glcm_contrast'] * feats['edge_density'] + feats['laplacian_var'] / 1000
    feats['hog_energy'] = feats['hog_mean']**2 + feats['hog_std']**2
    
    lbp_cols = [feats[f'lbp_{j}'] + eps for j in range(10)]
    feats['lbp_entropy'] = -np.sum(lbp_cols * np.log(lbp_cols))
    
    for k in feats:
        if np.isnan(feats[k]) or np.isinf(feats[k]):
            feats[k] = 0.0

    return feats

# --- Main App ---
st.title("🚁 Smart Drone Parking System")
st.markdown("Monitor parking slots in real-time, predict occupancy using machine learning, and analyse model performance.")

tab1, tab2, tab3 = st.tabs(["📸 Predict Slot", "📊 Model Comparison", "🚗 Synthetic Grid Demo"])

with tab1:
    st.header("Image Occupancy Predictor")
    
    if not model_artifact:
        st.warning(f"Model file `{MODEL_PATH}` not found! Please run the Jupyter Notebook to train and save the model.")
        st.stop()
        
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload Parking Slot Image", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Slot", use_container_width=True)
            
            # Predict
            img_np = np.array(image)
            raw_feats = extract_features(img_np)
            
            # Reconstruct the feature vector expected by the model
            feature_names = model_artifact['feature_names']
            scaler = model_artifact['scaler']
            model = model_artifact['model']
            threshold = model_artifact['threshold']
            
            # Fill missing contextual features with 0
            df_in = pd.DataFrame([raw_feats])
            for col in feature_names:
                if col not in df_in.columns:
                    df_in[col] = 0.0
            
            df_in = df_in[feature_names]
            
            # Scale
            X_scaled = scaler.transform(df_in)
            
            # Predict
            proba = model.predict_proba(X_scaled)[0, 1]
            is_occupied = proba >= threshold
            
    with col2:
        if uploaded_file is not None:
            st.subheader("Prediction Result")
            if is_occupied:
                st.error(f"**Status:** OCCUPIED")
            else:
                st.success(f"**Status:** FREE")
            
            st.metric("Confidence (Occupied Prob)", f"{proba * 100:.2f}%")
            
            st.subheader("Top Extracted Features")
            # Bar chart of top feature values
            f_values = df_in.iloc[0].abs().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(6, 4))
            f_values.plot(kind='barh', ax=ax, color='teal')
            ax.invert_yaxis()
            ax.set_title("Top 10 Feature Absolute Values (before scaling)")
            st.pyplot(fig)

with tab2:
    st.header("Model Performance Comparison")
    st.markdown("Comparison of baseline machine learning models against our champion XGBoost/LightGBM model.")
    
    # Hardcoded values from notebook
    results = {
        "Model": ["LightGBM", "Random Forest", "SVM (RBF)", "Logistic Regression", "XGBoost (Champion)"],
        "CV F1 Score": [0.957, 0.941, 0.910, 0.885, 0.961],
        "CV AUC-ROC": [0.985, 0.978, 0.955, 0.932, 0.991]
    }
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)
    
    st.info("The model artifacts are generated and saved dynamically by the Jupyter notebook at the end of its training pipeline.")

with tab3:
    st.header("Synthetic Parking Grid Demo")
    st.markdown("A 5x5 simulated parking lot demonstrating the real-time prediction layout.")
    
    if st.button("Generate Random Grid"):
        grid_data = np.random.choice([0, 1], size=(5, 5), p=[0.6, 0.4])
        
        for r in range(5):
            cols = st.columns(5)
            for c in range(5):
                with cols[c]:
                    # 1 = occupied (red), 0 = free (green)
                    if grid_data[r, c] == 1:
                        st.markdown(
                            "<div style='background-color: #E74C3C; color: white; padding: 20px; text-align: center; border-radius: 5px; font-weight: bold;'>Occupied</div>", 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            "<div style='background-color: #2ECC71; color: white; padding: 20px; text-align: center; border-radius: 5px; font-weight: bold;'>Free</div>", 
                            unsafe_allow_html=True
                        )
    else:
        st.write("Click the button above to generate a randomized occupancy grid.")

