# Import necessary libraries
import base64
import csv
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Function to save predictions in a CSV file
def save_predictions(filename, predictions):
    """
    Save predictions to a CSV file.
    :param filename: Name of the file to save predictions
    :param predictions: List of predictions to save
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Name', 'Predicted Class', 'Probability'])
        for pred in predictions:
            writer.writerow(pred)

# Function to create a download link for the saved file
def create_download_link(filename):
    """
    Create a download link for the given file.
    :param filename: Name of the file for which to create the download link
    :return: HTML href string for downloading the file
    """
    with open(filename, "rb") as file:
        # Encoding the file in base64
        encoded = base64.b64encode(file.read()).decode()
    # Creating the download link
    href = f'<a href="data:file/csv;base64,{encoded}" download="{filename}">Download predictions</a>'
    return href

# Main function to display the rice detector
def rice_detector(version='v3'):
    st.title("Rice Varieties Detector")
    
    st.markdown("""
        Upload images of individual rice grains here to identify their varieties. The model recognizes five types of rice: Arborio, Basmati, Ipsala, Jasmine, and Karacadag.
        """)
    
    st.markdown("""
        Please upload a clear image of an individual rice grain. 
        The grain should be centered and occupy a significant portion of the image, 
        similar to the reference image provided below.
        """)
    
    st.image(f'outputs_{version}/performance/Rice_example.jpg')  # path to reference image
    
    st.warning("You can upload no more than 4 images at a time.")

    # Load the trained model
    model_path = f'outputs_{version}/final_model/final_rice_model_2.keras'
    model = load_model(model_path)

    # Set the probability threshold to 85%
    probability_threshold = 85  

    # Limit the number of uploaded files
    max_files = 4
    uploaded_files = st.file_uploader("Choose an image...", accept_multiple_files=True, type=["jpg", "jpeg"])

    # Display the link to Kaggle dataset
    st.write("Examples of images can be taken from [here](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset).")

    # Initialize list to store predictions
    all_predictions = []

    if uploaded_files and len(uploaded_files) > max_files:
        # Use st.error to display the message with a red background
        st.error(f"You have uploaded {len(uploaded_files)} images. Please upload no more than {max_files} images. Delete excess images by clicking the 'x' icon next to them.")
    else:
        for uploaded_file in reversed(uploaded_files): # Iterate in reverse order
            if uploaded_file is not None:
                # Open the image
                image = Image.open(uploaded_file)
                # Reducing image size
                st.image(image, caption='Uploaded Image', width=250) 

                # Preprocess and classify the image
                processed_image = preprocess_image(image)
                predictions = model.predict(processed_image)

                # Identify the class with the highest probability
                classes = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag", "Non-Rice"]
                predicted_class = np.argmax(predictions, axis=1)[0]
                probability = np.max(predictions) * 100

                # Handling cases with low prediction confidence
                if probability < probability_threshold:
                    st.error("The model's confidence is low. The uploaded image may not be one of the target rice varieties or the image quality is not clear.")
                    all_predictions.append([uploaded_file.name, "Non-Rice or Unclear", "N/A"])
                elif classes[predicted_class] == "Non-Rice":
                    # If the image is recognized as non-rice by the model
                    st.error("The uploaded image is not recognized as one of the target rice varieties: Arborio, Basmati, Ipsala, Jasmine, or Karacadag. Please ensure the image is clear and the grain is properly focused.")
                    all_predictions.append([uploaded_file.name, "Non-Rice or Unclear", "N/A"])
                else:
                    # For high-confidence predictions
                    st.write(f"Prediction: {classes[predicted_class]} with probability {probability:.2f}%")
                    all_predictions.append([uploaded_file.name, classes[predicted_class], f'{probability:.2f}%'])
        # Save predictions and create a download link
        if all_predictions:
            predictions_filename = 'predictions.csv'
            save_predictions(predictions_filename, all_predictions)
            st.markdown(create_download_link(predictions_filename), unsafe_allow_html=True)

# Function to preprocess images for the model
def preprocess_image(image, target_size=(128, 128)):
    """
    Preprocess the image for the model.
    :param image: Image to preprocess
    :param target_size: Target size of the image
    :return: Preprocessed image array
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = keras_image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Call the function to display the detector
rice_detector()