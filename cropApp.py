from flask import Flask, request, jsonify
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Load model and processor only once at startup
MODEL_ID = "wambugu71/crop_leaf_diseases_vit"

# Initialize the model and processor
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)

@app.route('/predict', methods=['POST'])
def predict_disease():
    """
    Endpoint for crop leaf disease classification
    
    Accepts:
    - Image in base64 encoding
    - Or file upload with key 'image'
    
    Returns:
    - Predicted disease label and confidence score
    """
    try:
        image = None
        
        # Check if image is uploaded as file
        if 'image' in request.files:
            file = request.files['image']
            image = Image.open(file.stream).convert('RGB')
        
        # Check if image is sent as base64
        elif request.json and 'image' in request.json:
            image_data = base64.b64decode(request.json['image'])
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        else:
            return jsonify({
                'error': 'No image provided. Please upload an image file or provide a base64 encoded image.'
            }), 400
        
        # Preprocess and run inference
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get prediction probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get top prediction
            predicted_class_idx = logits.argmax(-1).item()
            confidence = probabilities[0][predicted_class_idx].item()
            
            # Get the label
            predicted_label = model.config.id2label[predicted_class_idx]
            
            return jsonify({
                'disease': predicted_label,
                'confidence': round(confidence * 100, 2),
                'success': True
            })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

if __name__ == '__main__':
    # For development only - use proper WSGI server in production
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    