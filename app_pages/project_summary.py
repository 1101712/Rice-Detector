import streamlit as st
import matplotlib.pyplot as plt


def page_summary():

    st.title("Project Summary")

    st.info(f"""
    ### **General Information**

    Rice, which is among the most widely produced grain products worldwide, has many genetic varieties. These varieties are separated from each other due to some of their features. These are usually features such as texture, shape, and color. With these features that distinguish rice varieties, it is possible to classify and evaluate the quality of seeds. In this study, Arborio, Basmati, Ipsala, Jasmine, and Karacadag, which are five different varieties of rice often grown in Turkey, were used.  
    
    It is also crucial to distinguish these varieties because they differ significantly in the characteristics of the product obtained after processing. Each variety yields a different texture, flavor, and level of stickiness when cooked. For instance, Arborio rice is known for its creamy and chewy texture, making it ideal for risotto, whereas Basmati is prized for its distinctive aroma and long, slender grains that become fluffy and separate upon cooking, perfect for biryani or pilaf dishes. Understanding these nuances is vital for culinary experts, chefs, and food enthusiasts who seek to optimize their recipes and dish presentation based on the rice type.  
    
    Additionally, distinguishing between these rice varieties is important for agricultural and economic reasons. Each variety may require different cultivation conditions, harvesting times, and care, impacting agricultural practices and yield. From a commercial perspective, accurate identification of rice types can aid in proper marketing and pricing strategies, ensuring that consumers receive the product they expect. Furthermore, in the realm of food science and nutrition, different rice types have varying nutritional profiles and health benefits, making their identification crucial for dietary planning and health-conscious eating.  
    
    Therefore, the development of an application that can classify rice varieties accurately is not only a technological achievement but also a tool that serves multiple practical purposes. It aids culinary professionals in selecting the right rice type for their dishes, assists farmers and agribusinesses in proper crop management, and helps consumers make informed choices about their food, aligning with their culinary preferences and health requirements.  

    ### **Project Dataset**  

    This dataset is a combination of two parts. The first part includes images of the specified rice varieties, and the second part comprises various miscellaneous images. The inclusion of 30,000 images, 5,000 from each class of rice grain and non-rice images, ensures a comprehensive dataset. All images have been uniformly processed and resized, then divided into three subsets for training, validation, and testing the model. The second dataset is crucial for training the model to recognize and differentiate not only between varieties of rice but also to identify and correctly handle images of non-rice subjects. This is essential to ensure that if an image of a different rice variety or an unrelated image is mistakenly uploaded by a user, the model can still accurately classify or flag it as non-rice.

        """)

    # Link to README file, so the users can have access to full project documentation
    st.write(f"""
    For additional information, please visit and **read** the [Project README file](https://github.com/1101712/Rice-Detector/).
        """)

    st.info(f"""
    ### **Business requirements**
    The project has 5 business requirements:  

    **Business Requirement 1: Visual Analysis of Rice Varieties**

    - Analyzing average and variability images for each rice variety to identify distinct features.
    - Investigating differences between the varieties through comparative studies.
    - Creating image montages to showcase the diversity within each variety, enhancing visual understanding.
    
    **Business Requirement 2: Machine Learning Model Development and Performance Metrics**

    - Developing a Neural Network-based ML system for precise classification, tailored to accurately identify each of the five rice varieties.
    - Addressing file size limitations for GitHub and deployment platforms, ensuring smooth operation and accessibility.
    - Setting performance goals, such as achieving an accuracy of over 95%, to meet the client's high standards for precision.
    - Implementing regular performance evaluations to maintain and improve the model's accuracy and efficiency over time.
    
    **Business Requirement 3: Reporting**

    - Generating detailed prediction reports for each analyzed rice image, providing insights into the classification results. These reports are crucial for quality control and product categorization, enhancing the client's decision-making process.
    
    **Business Requirement 4: Scalability and Adaptability**

    - Ensuring the model's scalability to handle large datasets efficiently without compromising performance.
    - Adapting the model for potential application to other grain types or similar classification tasks, increasing its utility.
    
    **Business Requirement 5: A user-friendly interface that allows for easy uploading of rice images and retrieval of prediction reports**

    - Create the Rice Varieties Detector with image upload and prediction report download capabilities.
        """)

