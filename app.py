import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Set confidence threshold for classification
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.69'))

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel

    def load_model():
        try:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            learned_prompts = torch.load(os.getenv('LEARNED_PROMPTS_PATH', 'src/models/learned_prompts.pt'), map_location=torch.device('cpu'))
            return model, processor, learned_prompts
        except Exception as e:
            print(f"Failed to load model components: {e}")
            return None, None, None

    model, processor, learned_prompts = load_model()

    def classify_image(image, model, processor, learned_prompts):
        try:
            inputs = processor(images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            similarities = torch.matmul(image_features, learned_prompts.t())
            probabilities = torch.nn.functional.softmax(similarities, dim=1)
            confidence, class_id = torch.max(probabilities, dim=1)
            confidence = confidence.item()
            class_id = class_id.item()

            if confidence > CONFIDENCE_THRESHOLD:
                if class_id == 0:
                    result = f"Cool! I'm pretty sure this is bread (confidence {confidence * 100:.2f}%)"
                else:
                    result = f"I'm quite sure this image is not bread (confidence: {confidence * 100:.2f}%)"
            else:
                result = f"I'm not confident enough how to classify this image (confidence: {confidence * 100:.2f}%)"

            return result
        except Exception as e:
            return f"Error during classification: {str(e)}"

except ImportError:
    print("Warning: Unable to import torch or transformers. Running in limited mode.")
    model, processor, learned_prompts = None, None, None

    def classify_image(image, model, processor, learned_prompts):
        return "Classification is currently unavailable."

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            if model and processor and learned_prompts is not None:
                result = classify_image(image, model, processor, learned_prompts)
                return jsonify({'result': result})
            else:
                return jsonify({'error': 'Model not available. Please check server logs.'})
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

