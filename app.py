from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
import tifffile as tiff
from PIL import Image
import io
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PREDICTIONS_FOLDER'] = 'static/predictions'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREDICTIONS_FOLDER'], exist_ok=True)

# Load Model
class UNetModel(nn.Module):
    def __init__(self, encoder):
        super(UNetModel, self).__init__()
        self.unet = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, x):
        return self.unet(x)

model = UNetModel(encoder="resnet18")
model.load_state_dict(torch.load("best_unet_model.pth", map_location=torch.device("cpu")))
model.eval()

def normalize_image(image):
    norm_img = np.zeros_like(image, dtype=np.float32)
    for b in range(image.shape[2]):
        band = image[:, :, b]
        min_val = np.min(band)
        max_val = np.max(band)
        if max_val > min_val:
            norm_img[:, :, b] = (band - min_val) / (max_val - min_val)  # Normalize to 0-1
        else:
            norm_img[:, :, b] = 0  
    return norm_img

def preprocess_image(image_path):
    # Read TIFF image
    tif_image = tiff.imread(image_path)
    # Extract RGB channels for display (Blue=1, Green=2, Red=3)
    rgb_image = tif_image[:, :, [3, 2, 1]]
    rgb_image = ((rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min()) * 255).astype(np.uint8)
    
    # Extract selected channels for model input (NIR=4, SWIR1=5, Green=2)
    selected_channels = tif_image[:, :, [4, 5, 2]].astype(np.float32)
    
    # Normalize selected channels
    normalized_image = normalize_image(selected_channels)
    
    # Convert to tensor
    image_tensor = torch.tensor(normalized_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return image_tensor, rgb_image

def save_mask(prediction, filename):
    mask = (np.array(prediction) * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask)
    mask_path = os.path.join(app.config['PREDICTIONS_FOLDER'], filename)
    mask_image.save(mask_path)
    return mask_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error='No file uploaded')
        
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)
        
        image_tensor, rgb_image = preprocess_image(image_path)

        with torch.no_grad():
            output = model(image_tensor)
            prediction = (torch.sigmoid(output).squeeze().numpy() > 0.5).astype(np.uint8)

        mask_path = save_mask(prediction, filename.replace('.tif', '_mask.png'))
        rgb_path = os.path.join(app.config['UPLOAD_FOLDER'], filename.replace('.tif', '_rgb.png'))
        Image.fromarray(rgb_image).save(rgb_path)

        return render_template('index.html', image_path=rgb_path, mask_path=mask_path)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
