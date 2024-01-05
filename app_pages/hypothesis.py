import streamlit as st
import matplotlib.pyplot as plt
import joblib

# Loading the model evaluation results
evaluation = joblib.load("outputs/v2/evaluation.pkl")

def hypothesis():
    st.title("Project Hypothesis")

    st.write("### Hypothesis 1 and Validation")

    st.success(
        "Distinct distinguishing features can be identified in images of five rice varieties: Arborio, Basmati, Ipsala, Jasmine, and Karacadag."
    )
    st.info(
        "The study involved examining images of different rice varieties provided by the client. "
        "Each variety has unique characteristics in terms of shape, size, and texture."
    )
    st.write("To explore these differences, visit the Rice Varieties Visualizer tab.")

    st.warning(
        "The model was trained to recognize these distinct features, enabling it to differentiate between the rice varieties accurately."
        " This demonstrates the model's ability to learn and generalize based on the distinguishing characteristics of each variety."
    )

    st.write("### Hypothesis 2 and Validation")

    st.success(
        "High accuracy in classifying rice varieties can be achieved with a large dataset and deep learning."
    )
    st.info(
        "The project's model was trained on a dataset comprising approximately 15,000 images for each rice variety. "
        "This extensive dataset, combined with deep learning techniques, aimed to achieve high classification accuracy."
    )
    st.warning(
        "The final model achieved an accuracy of 99.30%, validating the hypothesis. "
        "This underscores the effectiveness of using a large dataset and deep learning for precise image classification."
    )

    # Visualization of Confusion Matrix
    st.write("### Confusion Matrix")
    st.write("This graph shows how often the model correctly or incorrectly classifies each class of rice.")
    st.write("The Y-axis (Actual) displays the true class labels, while the X-axis (Predicted) shows the model's predictions.")
    st.write("Diagonal values represent the number of correct predictions the model makes for each class, while non-diagonal values show instances of misclassification.")
    confusion_matrix_img = plt.imread("outputs/v2/performance/confusion_matrix.png")
    st.image(confusion_matrix_img, caption='Confusion Matrix')

    # Visualization of Classification Report
    st.write("### Classification Report")
    st.write("This graph presents key classification metrics for each class, including precision, recall, and F1-score.")
    st.write("Precision indicates the percentage of predictions for a class that were correct.")
    st.write("Recall reflects the percentage of actual examples of a class that were correctly identified.")
    st.write("F1-score is the harmonic mean of precision and recall, providing an overall measure of classification accuracy.")
    classification_report_img = plt.imread("outputs/v2/performance/classification_report.png")
    st.image(classification_report_img, caption='Classification Report')

    # Visualization of ROC Curve
    st.write("### ROC Curve")
    st.write("This graph represents the Receiver Operating Characteristic (ROC) curves for each class.")
    st.write("The X-axis shows the False Positive Rate, while the Y-axis shows the True Positive Rate.")
    st.write("An ideal model aspires to a point in the top-left corner, where the True Positive Rate is 1 and the False Positive Rate is 0.")
    st.write("The Area Under the Curve (AUC) serves as a measure of the model's overall ability to distinguish between classes, where 1.0 is the perfect score.")
    roc_curve_img = plt.imread("outputs/v2/performance/roc_curve.png")
    st.image(roc_curve_img, caption='ROC Curve for Rice Varieties')

    st.write(
        "For additional information, please visit and **read** the "
        "[Project README file](https://github.com/1101712/Rice-Detector)."
    )

# Call the function to display the hypothesis page
hypothesis()