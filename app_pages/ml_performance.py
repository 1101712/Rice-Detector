import streamlit as st
import os

def ml_performance():
    st.title("Machine Learning Performance Metrics")
    
    st.markdown("""
        This section of the dashboard presents detailed performance metrics of the machine learning model used to classify rice varieties. It includes the distribution of image labels across the training, validation, and test sets, as well as a summary of the model's predictive accuracy. For visualizations of the Confusion Matrix, Classification Report, and ROC Curve, please refer to the _Project Hypothesis_ page.
        """)

    # Dataset Distribution
    st.write("### Dataset Percentage Distribution")
    st.markdown("""
    The pie chart breaks down the dataset into training, validation, and test sets by percentage, showing us the proportion of images used for each phase of the model's development.

    Train set (70% of the whole dataset) is the initial data used to 'fit' the model which will learn on this set how to generalize and make prediction on new unseen data.

    Validation set (15% of the dataset) helps to improve the model performance by fine-tuning the model after each epoch (one complete pass of the training set through the model).

    The test set (15% of the dataset) informs us about the final accuracy of the model after completing the training phase. It's a batch of data the model has never seen.
    """)
    set_distribution_path = os.path.join('outputs', 'v2', 'performance', 'set_distribution.png')
    if os.path.exists(set_distribution_path):
        st.image(set_distribution_path, caption='Set Distribution in Dataset')

     # Label Frequencies
    st.write("### Label Frequencies for Train, Validation, and Test Sets")
    st.markdown("""
    This bar chart shows the count of images for each rice variety within the training, validation, and test sets, helping us to ensure that the model is trained, validated, and tested on a balanced number of images from each category.
    """)
    label_distribution_path = os.path.join('outputs', 'v2', 'performance', 'label_distribution.png')
    if os.path.exists(label_distribution_path):
        st.image(label_distribution_path, caption='Label Distribution in Each Set')

    # Color Distribution
    st.write("### Color Distribution Across Varieties")
    st.markdown("""
    These histograms illustrate the average distribution of red, green, and blue color intensities across all rice images, potentially highlighting unique color features of different rice varieties.
    """)
    color_distribution_path = os.path.join('outputs', 'v2', 'performance', 'color_distribution.png')
    if os.path.exists(color_distribution_path):
        st.image(color_distribution_path, caption='Color Distribution')

    # Model Accuracy
    st.write("### Model Accuracy Over Epochs")
    st.markdown("""
    This line chart tracks the model's accuracy and loss at each epoch during training. A declining loss and rising accuracy indicate the model's improving performance over time.
    Loss represents the cumulative mistakes the model makes for each example during the training (referred to as 'loss') and validation phases (referred to as 'val_loss').

The value of loss indicates the model's performance after every optimization cycle — a lower loss suggests better performance.

Accuracy measures the precision of the model's predictions ('accuracy') against the actual outcomes ('val_accuracy').

A well-performing model on new, unseen data demonstrates its ability to generalize, indicating that it has learned the underlying patterns rather than just memorizing the training data.
    """)
    model_accuracy_chart_path = os.path.join('outputs', 'v2', 'performance', 'combined_loss_accuracy_plot.png')
    if os.path.exists(model_accuracy_chart_path):
        st.image(model_accuracy_chart_path, caption='Model Accuracy Chart')

    st.markdown("""
        ### Additional Visualizations
        For further details on the model's performance and visual assessments such as the Confusion Matrix, Classification Report, and ROC Curve, please visit the **Project Hypothesis** page within this dashboard.
        """)

    st.markdown("""
        ### More Information
        For a comprehensive understanding of the project, including the methodology, dataset, and conclusions, please refer to the [Project README file](https://github.com/1101712/Rice-Detector).
        """)

# Call the function to display the performance metrics page
ml_performance()