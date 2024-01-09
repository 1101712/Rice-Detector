import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image


# Class to generate multiple Streamlit pages using an object oriented approach
class MultiPage:

    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name
     
    def add_page(self, title, func) -> None:
        self.pages.append({"title": title, "function": func})

    def run(self):
        # add logo
        logo_image = Image.open('outputs_v3/performance/logo.jpg')
        st.sidebar.image(logo_image, use_column_width=True)
        st.sidebar.markdown("""
            <h1 style='text-align: center; color: blue; font-size: 34px; font-weight: bold; font-family: Arial, sans-serif;'>Rice Varieties Detector</h1>
            """, unsafe_allow_html=True)

        st.sidebar.markdown("## Settings")
        view_mode = st.sidebar.radio("View Mode", ["Expert", "Quick Access"])

        if view_mode == "Expert":
            available_pages = self.pages
        else:
            # In Quick Access mode, show only key pages
            available_pages = [page for page in self.pages if page["title"] in ["Rice Varieties Detector", "Project Summary"]]

        selected_page = st.sidebar.radio('Menu', available_pages, format_func=lambda page: page['title'])
        selected_page['function']()
    