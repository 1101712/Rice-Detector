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

    A total of 75,000 grain images, 15,000 from each of these varieties, are included in the dataset.
        """)

    # Link to README file, so the users can have access to full project documentation
    st.write(f"""
    For additional information, please visit and **read** the [Project README file](https://github.com/1101712/Rice-Detector/).
        """)

    st.info(f"""
    ### **Business requirements**
    The project has 2 business requirements:  

    - The client is interested in conducting a study to visually differentiate five rice varieties: Arborio, Basmati, Ipsala, Jasmine, and Karacadag.  

    - The client seeks to determine the specific variety of rice depicted in the uploaded images.  

    - Download a prediction report of the examined rice image.
        """)

