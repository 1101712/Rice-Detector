import streamlit as st
import matplotlib.pyplot as plt


# Class to generate multiple Streamlit pages using an object oriented approach
class MultiPage:

    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name
     
    def add_page(self, title, func) -> None:
        self.pages.append({"title": title, "function": func})

    def run(self):
        st.sidebar.title("Settings")
        view_mode = st.sidebar.radio("View Mode", ["Standard", "Quick Access"])

        if view_mode == "Standard":
            available_pages = self.pages
        else:
            # In Quick Access mode, show only key pages
            available_pages = [page for page in self.pages if page["title"] in ["Rice Varieties Detector", "Project Summary"]]

        selected_page = st.sidebar.radio('Menu', available_pages, format_func=lambda page: page['title'])
        selected_page['function']()
    