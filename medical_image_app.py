import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load pre-trained model
model = tf.keras.models.load_model('lung_cancer_model.h5')

# Define class labels
class_labels = ['Normal', 'Lung Cancer']

# Function to preprocess image
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Main Streamlit app
def main():
    st.title("Lung Cancer Detection from Chest X-ray")
    st.write("Upload a chest X-ray image to classify if it has signs of lung cancer.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")

        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]

        st.write(f"**Predicted Class:** {class_labels[predicted_class]}")
        st.write(f"**Confidence:** {confidence:.2f}")

if __name__ == "__main__":
    main()
