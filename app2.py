import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import streamlit as st
from model_classes import ResNet9

# Function to predict the category
def predict_image(img, model, classes):
    """Converts image to array and returns the predicted class
    with the highest probability"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()])

    img = transform(img)
    xb = img.unsqueeze(0)

    device = torch.device('cpu')
    model.to(device)

    xb = xb.to(device)
    yb = model(xb)

    yb = yb.cpu()
    _, preds = torch.max(yb, dim=1)

    predicted_class = classes[preds[0].item()]

    return predicted_class

# Home Page
def home():
    # Page content
    st.title('Plant Disease Detection - Home Page')
    st.write('Welcome to our Plant Disease Detection application!')

    # Information about plant disease
    st.header('Information about Plant Disease')
    st.write('Plant pathogens affecting field crops have great economic importance and warrant widespread and frequent use of pesticides. Besides more common ways of infection of the crop plants, pathogens and weed seeds may also be introduced by agricultural use of BW. The inactivation kinetics of plant pathogens in different kinds of hygiene treatment of BW is, in general, the same as that for human and animal pathogens. However, fungal pathogens may develop resting spores, await more advantageous conditions, and survive for many years in, for example, the soil. They are also rather resistant to a number of treatment methods. Some of them may also use many different plant species as hosts. Examples of long-living fungus include white mould (Scelotinia sclerotiorum) and clubroot of crucifers (Plasmodiophora brassicae). Furthermore, some plant viruses are hard to be inactivated using heat, for example, tobacco mosaic virus.')

    # Importance of plant disease
    st.header('Importance of Plant Disease Prediction ')
    st.write('lant disease prediction plays a crucial role in agriculture and crop management. Here are some key points highlighting the importance of plant disease prediction:')
    st.write('1. Early detection: Plant diseases can cause significant damage to crops if not detected and managed early. By predicting diseases in their early stages, farmers can take proactive measures to prevent the spread and minimize the impact on crop yield.')
    st.write('2. Yield optimization: Plant diseases can significantly reduce crop yields, leading to economic losses for farmers. By predicting diseases, farmers can implement timely interventions such as targeted treatments, optimized irrigation, and appropriate use of fertilizers, thereby maximizing crop yields.')
    st.write('3. Resource management: Accurate disease prediction helps farmers optimize the use of resources such as pesticides, water, and fertilizers. Instead of applying treatments uniformly across entire fields, farmers can focus on affected areas, reducing the overall use of chemicals and minimizing environmental impact.')
    st.write('4. Cost reduction: Early disease detection and targeted interventions can help reduce the cost of crop protection. By preventing widespread infections and implementing appropriate management strategies, farmers can save money on excessive pesticide applications and reduce crop losses.' )
    st.write('5. Sustainable farming practices: Plant disease prediction promotes sustainable farming practices by minimizing the reliance on chemical treatments. By accurately identifying diseases, farmers can adopt integrated pest management approaches, including biological control methods and cultural practices, to maintain a healthy balance in the ecosystem.')
    st.write('6. Food security: With a growing global population, ensuring food security is a top priority. By effectively predicting and managing plant diseases, agricultural productivity can be maintained, ensuring a stable supply of food to meet the increasing demand.')
    st.write('7. Decision support system: Plant disease prediction models provide valuable information to farmers and agricultural stakeholders. They serve as decision support systems, enabling farmers to make informed choices regarding disease management, crop rotation, planting schedules, and other agricultural practices.')
    
    # Number of classes
    st.header('Number of Classes')
    st.write('Our model is trained to predict 38 different classes of plant diseases.')

    # Button to go to prediction page
    if st.button('Let\'s Predict the Disease'):
        st.session_state.page = 'Prediction'

# Prediction Page
def prediction():
    # Load the model
    model = torch.load('plant-disease-fullmodel.pth', map_location=torch.device('cpu'))

    # Load the classes
    with open('plant_disease_classes.pkl', 'rb') as f:
        classes = pickle.load(f)

    # Page content
    st.title('Plant Disease Detection - Prediction Page')
    st.write('Upload an image to predict the plant disease.')

    # Image upload
    uploaded_image = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    # Perform prediction if image is uploaded
    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        # Call the predict_image() function to get the predicted category
        predicted_category = predict_image(image, model, classes)

        # Display the predicted category and image
        st.header('Predicted Category:')
        st.write(predicted_category)
        st.image(image, caption='Uploaded Image')

    # Button to go back to home page
    if st.button('Go Back to Home'):
        st.session_state.page = 'Home'

# Main App
def main():
    # Set page configuration
    st.set_page_config(page_title='Plant Disease Detection', page_icon='🌱')

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'

    # Render the selected page
    if st.session_state.page == 'Home':
        home()
    elif st.session_state.page == 'Prediction':
        prediction()

if __name__ == '__main__':
    main()
