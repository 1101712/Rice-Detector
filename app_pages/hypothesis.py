import streamlit as st
import matplotlib.pyplot as plt
import joblib

# Loading the model evaluation results
evaluation = joblib.load("outputs_v3/evaluation.pkl")

def hypothesis(version='v3'):
    st.title("Project Hypothesis")

    st.write("### Hypothesis 1 and Validation")

    st.success(
        "Distinct features can be identified in images of five rice varieties and non-rice objects."
    )
    st.info(
        "The analysis involved examining images of rice varieties and non-rice objects to identify unique characteristics. "
        "The varieties and non-rice objects have distinctive features such as shape, size, and texture."
    )
    st.write("To explore these differences, visit the Rice Varieties Visualizer tab.")

    st.warning(
        "The model was trained not only to recognize the distinct features of each rice variety but also to accurately identify non-rice objects. "
        "This demonstrates the model's capability to learn and generalize from a comprehensive range of visual characteristics."
    )

    st.write("### Hypothesis 2 and Validation")

    st.success(
        "A combined dataset of rice varieties and non-rice images enhances the model's predictive accuracy."
    )
    st.info(
        "The model was trained on a robust dataset that included diverse images of rice and non-rice items. "
        "The large and varied dataset, along with advanced deep learning techniques, was utilized to refine classification accuracy."
    )
    st.warning(
        "The final model, incorporating non-rice image recognition, achieved an accuracy rate exceeding 99.47%, validating the hypothesis. "
        "This result highlights the significance of a diversified dataset and sophisticated algorithms in achieving precise image classification."
    )

    # Visualization of Confusion Matrix
    st.write("### Confusion Matrix")
    st.write("This graph illustrates the frequency of correct and incorrect classifications for each class, including non-rice.")
    st.write("The Y-axis (Actual) displays the true class labels, while the X-axis (Predicted) shows the model's predictions.")
    st.write("Diagonal values represent the number of correct predictions the model makes for each class, while non-diagonal values show instances of misclassification.")
    confusion_matrix_img = plt.imread(f"outputs_{version}/performance/confusion_matrix_2.png")
    st.image(confusion_matrix_img, caption='Confusion Matrix')

    # Visualization of Classification Report
    st.write("### Classification Report")
    st.write("This graph presents key classification metrics for each class, including precision, recall, and F1-score.")
    st.write("Precision indicates the percentage of predictions for a class that were correct.")
    st.write("Recall reflects the percentage of actual examples of a class that were correctly identified.")
    st.write("F1-score is the harmonic mean of precision and recall, providing an overall measure of classification accuracy.")
    classification_report_img = plt.imread(f"outputs_{version}/performance/classification_report_2.png")
    st.image(classification_report_img, caption='Classification Report')

    # Visualization of ROC Curve
    st.write("### ROC Curve")
    st.write("This graph represents the Receiver Operating Characteristic (ROC) curves for each class.")
    st.write("The X-axis shows the False Positive Rate, while the Y-axis shows the True Positive Rate.")
    st.write("An ideal model aspires to a point in the top-left corner, where the True Positive Rate is 1 and the False Positive Rate is 0.")
    st.write("The Area Under the Curve (AUC) serves as a measure of the model's overall ability to distinguish between classes, where 1.0 is the perfect score.")
    roc_curve_img = plt.imread(f"outputs_{version}/performance/roc_curve_2.png")
    st.image(roc_curve_img, caption='ROC Curve for Rice Varieties')

    st.write(
        "For additional information, please visit and **read** the "
        "[Project README file](https://github.com/1101712/Rice-Detector)."
    )

# Call the function to display the hypothesis page
hypothesis()