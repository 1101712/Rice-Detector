import streamlit as st
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image

def rice_visualizer():
    st.title("Rice Varieties Visualizer")
    st.markdown(f"""
        Welcome to the Rice Varieties Visualizer, a key component of our dashboard that highlights the distinctive visual characteristics of different rice varieties. This interactive section is dedicated to showcasing the unique attributes of five specific rice types: Arborio, Basmati, Ipsala, Jasmine, and Karacadag.
        
        This visual information forms the foundation upon which our machine learning model is trained to classify the rice grains accurately.
        """)

    version = 'v2'

    rice_varieties = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

    # Title and expander for the 'Average and Variability' section
    st.markdown("### Difference between average and variability image")
    st.info(f"""
        In this part of the visualizer, you can explore the average appearance and variability of each rice variety. By selecting the checkboxes, you will see images that represent the average color, shape, and texture of each rice type, along with their variability within the dataset. These images are crucial for understanding the subtle yet important differences between each variety, providing a visual basis for the machine learning model's classification decisions.
        """)
    with st.expander("Click here to expand"):
        for variety in rice_varieties:
            if st.checkbox(f"Show {variety} Average and Variability", key=f"avg_var_{variety}"):
                avg_img_path = f"outputs/{version}/average_images/{variety}_average_variability.png"
                if os.path.exists(avg_img_path):
                    avg_image = Image.open(avg_img_path)
                    st.image(avg_image, caption=f'{variety} - Average and Variability')
                    st.write("---")
                else:
                    st.error(f"File not found: {avg_img_path}")

    # Title and expander for the 'Image Montage' section
    st.markdown("### Image Montage")
    st.info(f"""
        The montage section offers a closer look at each rice variety through a collection of individual grain images. This visual montage allows you to observe the natural variation within each type of rice, showcasing the diversity in size, shape, and color. It's an effective way to visually compare and contrast the rice varieties at a glance.
        """)
    with st.expander("Click here to expand"):
        for variety in rice_varieties:
            if st.checkbox(f"Show Montage for {variety}", key=f"montage_{variety}"):
                montage_img_path = f"outputs/{version}/montage/montage_{variety}.png"
                if os.path.exists(montage_img_path):
                    montage_image = Image.open(montage_img_path)
                    st.image(montage_image, caption=f'{variety} Montage')
                    st.write("---")
                else:
                    st.error(f"File not found: {montage_img_path}")
                    
    st.write("""
    For additional information, please visit and **read** the [Project README file](https://github.com/1101712/Rice-Detector/).
        """)

# Call the function to display the visualizer
rice_visualizer()

