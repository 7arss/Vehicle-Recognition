import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import pandas as pd
from keras.models import model_from_json

# Load class names from the names.csv file
names_df = pd.read_csv('names.csv')
class_names = names_df.set_index('class_id')['class_name'].to_dict()

# Load the model architecture from the JSON file
with open('car_classifier_model.json', 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

model.load_weights('car_classifier_model_weights.h5')  # Load the model weights from the H5 file
transform = transforms.Compose([
    transforms.ToTensor()
])

def predict_car_details(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        predictions = model(image)

    # extract labels and scores
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    return labels, scores

def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # open the image and create a thumbnail
        image = Image.open(file_path)
        image.thumbnail((150, 150))  # resize image so it displays
        img_display = ImageTk.PhotoImage(image)

        img_label.configure(image=img_display)
        img_label.image = img_display

        labels, scores = predict_car_details(file_path)
        for label, score in zip(labels, scores):
            if score >= 0.80:  # Consider predictions with high confidence
                car_name = class_names.get(label, 'Unknown')
                prediction_text = f"Predicted Car: {car_name}"
                result_label.config(text=prediction_text)
                break

# main UI window
root = tk.Tk()
root.title("Car Classifier")
root.geometry("400x400")
root.resizable(False, False)  # Lock the window size

frame = tk.Frame(root)
frame.pack(pady=20)

button = tk.Button(frame, text="Browse Image", command=browse_image)
button.pack(pady=10)

# Label to display the selected image
img_label = tk.Label(frame)
img_label.pack(pady=10)

# Label to display the prediction result
result_label = tk.Label(frame, text="Predicted Car:")
result_label.pack(pady=10)

root.mainloop()
