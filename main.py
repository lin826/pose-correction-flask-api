import argparse
import torch
import os

from flask import Flask, abort, request
from werkzeug.utils import secure_filename
from analyzer.vibe import Vibe

import uuid
import threading
import json

TMP_FOLDER = "tmp"
FILE_TAG = 'tmp_file'
ALLOWED_EXTENSIONS = {'mp4'} # {'png', 'jpg', 'jpeg', 'mp4'}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # mps is not supported by mmpose

app = Flask(__name__)
vibe_analyzer = Vibe(DEVICE)
semaphore = None # Will be initialized in __main__
os.makedirs(TMP_FOLDER, exist_ok=True)

def allowed_file(filename: str) -> bool:
    if '.' not in filename:
        return False
    return filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_result(filename: str):
    global semaphore

    base_dir = os.path.dirname(filename)
    img_folder = os.path.join(base_dir, f"{filename}_images")
    os.makedirs(img_folder)
    
    with semaphore:
        return vibe_analyzer.analyze(filename, img_folder)

def get_serial_id():
    return str(uuid.uuid4())

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if FILE_TAG not in request.files:
        abort(400, description="No specific file in the form data.")

    f = request.files[FILE_TAG]

    # Process the valid file
    if f and allowed_file(f.filename):
        filename = secure_filename(f.filename)
        file_save_path = os.path.join(TMP_FOLDER, get_serial_id())
        f.save(file_save_path)
        result = get_result(file_save_path)
        return json.loads('{"pose": "%s", "result": "%s"}' % (result[0], result[1]))
    
    abort(415, description="Not supported type.") # 415 Unsupported Media Type

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pose Correction Flask API")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='API host (str)')
    parser.add_argument('--port', type=int, default=8000, help='API port (int)')
    parser.add_argument('--size', type=int, default=3, help='Number of parallel processes (int)')
    args = parser.parse_args()
    
    semaphore = threading.Semaphore(args.size)
    app.run(host=args.host, port=args.port)
