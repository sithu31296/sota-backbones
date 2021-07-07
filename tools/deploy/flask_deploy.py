from PIL import Image
from flask import Flask, json, jsonify, request
from torchvision import transforms


def transform_image(infile):
    input_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(infile)
    return input_transforms(image).unsqueeze(0)

def get_prediction(model, input_tensor):
    outputs = model(input_tensor)
    _, y_hat = outputs.max(1)
    return y_hat.item()

def render_prediction(prediction_idx):
    class_name = 'Unknown'
    if img_class_map is not None:
        class_name = img_class_map[prediction_idx]
    return class_name


app = Flask(__name__)
model = 'model'
model.eval()

img_class_map = None

@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg': 'Try POSTing to the /predict endpoint with an RGB image attachment'})

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        input_tensor = transform_image(file)
        prediction_idx = get_prediction(model, input_tensor)
        class_name = render_prediction(prediction_idx)
        return jsonify({'class_id': prediction_idx, 'class_name': class_name})

if __name__ == '__main__':
    app.run()
    """
    To start the server from the shell, run the following
        FLASK_APP=app.py flask run

    Once the server is running, open another terminal window and test your new inference server
        curl -X POST -H "Content-Type: multipart/form-data" http://localhost:5000/predict -F "file=@kitten.jpg"

    If everything works correctly, you should receive a response similar to the following:
        {"class_id": ???, "class_name": "????"}
    """