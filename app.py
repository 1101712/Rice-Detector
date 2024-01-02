import streamlit as st
from app_pages.multipage import MultiPage

# Load pages scripts
from app_pages.project_summary import page_summary
from app_pages.rice_visualizer import rice_visualizer
from app_pages.rice_detector import rice_detector
from app_pages.hypothesis import hypothesis
from app_pages.ml_performance import ml_performance

# Create an instance of the app
app = MultiPage(app_name="Rice Varieties Detector")

# Add pages to the app
app.add_page("Project Summary", page_summary)
app.add_page("Rice Varieties Visualiser", rice_visualizer)
app.add_page("Rice Varieties Detector", rice_detector)
app.add_page("Project Hypothesis", hypothesis)
app.add_page("ML Performance Metrics", ml_performance)

# Run the app
app.run()
