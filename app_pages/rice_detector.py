import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

def rice_detector():
    st.title("Rice Varieties Detector")
    st.markdown("""
        Upload images of individual rice grains here to identify their varieties. The model recognizes five types of rice: Arborio, Basmati, Ipsala, Jasmine, and Karacadag.
        """)
    
    st.warning("You can upload no more than 4 images at a time.")

    # Loading the trained model
    model_path = 'outputs/v2/final_model/final_rice_model.keras'
    model = load_model(model_path)

    # Limiting the number of uploaded files
    max_files = 4
    uploaded_files = st.file_uploader("Choose an image...", accept_multiple_files=True, type=["jpg", "jpeg"])

    # Display the link to Kaggle dataset right after the file uploader
    st.write("Examples of images can be taken from [here](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset).")

    if len(uploaded_files) > max_files:
        st.warning(f"Please upload no more than {max_files} images.")
    else:
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)

                # Image preprocessing and classification
                processed_image = preprocess_image(image)
                predictions = model.predict(processed_image)

                # Identifying the class with the highest probability
                predicted_class = np.argmax(predictions, axis=1)[0]
                probability = np.max(predictions) * 100

                # Recognized rice varieties
                classes = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
                st.write(f"Prediction: {classes[predicted_class]} with probability {probability:.2f}%")
                st.write("---")
    
# Preprocessing the image for the model
def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = keras_image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Call the function to display the detector
rice_detector()

