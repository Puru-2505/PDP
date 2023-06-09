import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from model_classes import ResNet9
import pickle

def predict_image(img, model, classes):
    """Converts image to array and returns the predicted class
    with the highest probability"""
    # Resize and normalize the image
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()])

    img = transform(img)
    
    # Convert to a batch of 1
    xb = img.unsqueeze(0)
    
    # Move the model to CPU
    device = torch.device('cpu')
    model.to(device)
    
    # Get predictions from the model
    xb = xb.to(device)
    yb = model(xb)
    
    # Move predictions back to CPU
    yb = yb.cpu()
    
    # Pick the index with the highest probability
    _, preds = torch.max(yb, dim=1)
    
    # Retrieve the class label
    predicted_class = classes[preds[0].item()]
    
    # Display the image with the predicted category
    plt.imshow(img.permute(1, 2, 0))
    plt.axis('off')
    plt.title(f'Predicted Category: {predicted_class}')
    plt.show()
    
    return predicted_class

# Load the model
model = torch.load('plant-disease-fullmodel.pth', map_location=torch.device('cpu'))

# Load the classes
with open('plant_disease_classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# Load and preprocess the input image
input_image1 = Image.open('F:\plant_disease_files-20230608T181130Z-001\plant_disease_files\AppleScab2.jpg')
input_image2 = Image.open('F:\plant_disease_files-20230608T181130Z-001\plant_disease_files\PotatoEarlyBlight1.jpg')

# Predict the category and display the image
predicted_category = predict_image(input_image1, model, classes)

print('Predicted Category:', predicted_category)