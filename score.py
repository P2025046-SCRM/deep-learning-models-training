import os
import io
import json
import base64
import time
import datetime
import numpy as np
import tensorflow as tf
import uuid
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image
from pydantic import BaseModel, ValidationError
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

# --- 1. Global Variables ---
model_l1 = None
model_l2 = None

blob_service_client = None

# --- 2. Constants ---
IMAGE_SIZE = (299, 299)
L1_CLASS_NAMES = ["NoReciclable", "Reciclable"]
L2_CLASS_NAMES = ["Biomasa", "Metales", "Plastico", "Retazos"]

STORAGE_ACCOUNT_NAME = "scrmmldeployme8359139872"
CONTAINER_NAME = "processed-images-wfwaste"

# --- 3. Pydantic Input Schema ---
class ImageInput(BaseModel):
    image_base64: str

# --- 4. The init() Function ---
def init():
    """
    Called once when the server starts. Loads both models.
    """
    global model_l1, model_l2, blob_service_client
    try:
        model_dir = os.getenv("AZUREML_MODEL_DIR")
        if not model_dir:
            raise ValueError("AZUREML_MODEL_DIR environment variable not set")
        
        base_path = os.path.join(model_dir, "package_v1")

        model_l1_path = os.path.join(base_path, "1stLayerClassification", "1", "binary_inceptionv3_model_ft_1.keras")
        model_l2_path = os.path.join(base_path, "2ndLayerClassification", "1", "inceptionv3_model_1.keras")

        if not os.path.exists(model_l1_path) or not os.path.exists(model_l2_path):
            raise FileNotFoundError(f"Model files not found. Searched for: {model_l1_path} and {model_l2_path}")

        print(f"Model mount directory: {model_dir}")
        print(f"Base path (corrected): {base_path}")
        print(f"Attempting to load Model 1 from: {model_l1_path}")
        model_l1 = tf.keras.models.load_model(model_l1_path)
        print("Model 1 loaded successfully.")

        print(f"Attempting to load Model 2 from: {model_l2_path}")
        model_l2 = tf.keras.models.load_model(model_l2_path)
        print("Model 2 loaded successfully.")
        
        # DefaultAzureCredential will automatically use the Managed Identity
        blob_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(blob_url, credential=credential)
        print(f"BlobServiceClient initialized using Managed Identity for {STORAGE_ACCOUNT_NAME}.")
        
        print("--- All models loaded. Server is ready. ---")

    except Exception as e:
        print(f"FATAL ERROR in init(): {str(e)}")
        raise

# --- 5. Blob Upload Function ---
def upload_image_to_blob(image_bytes: bytes, blob_name: str) -> str:
    """
    Uploads image bytes to Azure Blob Storage and returns the permanent Base URL.
    """
    global blob_service_client
    if not blob_service_client:
        raise ConnectionError("BlobServiceClient is not initialized.")
        
    try:
        blob_client = blob_service_client.get_blob_client(
            container=CONTAINER_NAME, 
            blob=blob_name
        )
        
        # Upload the data
        blob_client.upload_blob(image_bytes, overwrite=True)
        
        # Construct the permanent Base URL
        base_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER_NAME}/{blob_name}"
        
        print(f"Image uploaded successfully. Base URL: {base_url}")
        return base_url
        
    except Exception as e:
        print(f"FATAL ERROR during blob upload: {e}") 
        raise 

# --- 6. Preprocessing & Helper Functions ---
def decode_image(base64_string: str) -> bytes:
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        return base64.b64decode(base64_string)
    except Exception as e:
        raise ValueError(f"Invalid base64 string: {str(e)}")

def load_and_prepare_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize(IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error in image preparation: {e}")
        raise

# --- 7. Prediction Logic ---

def run_layer1(data: ImageInput):
    image_bytes = decode_image(data.image_base64)
    prepared_img = load_and_prepare_image(image_bytes)
    prediction = model_l1.predict(prepared_img)
    score = float(prediction[0][0])
    if score > 0.5:
        prediction_name = L1_CLASS_NAMES[1]
        confidence = score
    else:
        prediction_name = L1_CLASS_NAMES[0]
        confidence = 1.0 - score
    return {"model": "layer1", "prediction": prediction_name, "confidence": confidence, "raw_score": score}

def run_layer2(data: ImageInput):
    image_bytes = decode_image(data.image_base64)
    processed_img = load_and_prepare_image(image_bytes)
    prediction = model_l2.predict(processed_img)
    pred_class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    return {"model": "layer2", "prediction": L2_CLASS_NAMES[pred_class_index], "confidence": confidence, "raw_output": prediction.tolist()}

def run_prod(data: ImageInput):
    start_total_time = time.perf_counter()
    classification_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    #decode and prepare img
    image_bytes = decode_image(data.image_base64)
    processed_img = load_and_prepare_image(image_bytes)
    
    #start L1
    start_l1_inference = time.perf_counter()
    prediction_l1 = model_l1.predict(processed_img)
    end_l1_inference = time.perf_counter()
    l1_inference_time = end_l1_inference - start_l1_inference
    score_l1 = float(prediction_l1[0][0])
    if score_l1 > 0.5:
        l1_prediction_name = L1_CLASS_NAMES[1]
        l1_confidence = score_l1
    else:
        l1_prediction_name = L1_CLASS_NAMES[0]
        l1_confidence = 1.0 - score_l1
    result_l1 = {"model": "layer1", "prediction": l1_prediction_name, "confidence": l1_confidence}
    
    #start L2
    result_l2 = None
    l2_inference_time = None
    if l1_prediction_name == "Reciclable":
        start_l2_inference = time.perf_counter()
        prediction_l2 = model_l2.predict(processed_img) 
        end_l2_inference = time.perf_counter()
        l2_inference_time = end_l2_inference - start_l2_inference
        l2_pred_index = np.argmax(prediction_l2)
        l2_prediction = L2_CLASS_NAMES[l2_pred_index]
        l2_confidence = float(np.max(prediction_l2))
        raw_score_list = prediction_l2[0].tolist()
        result_l2 = {"model": "layer2", "prediction": l2_prediction, "confidence": l2_confidence, "raw_score":raw_score_list}
    
    #create unique blob name
    timestamp_slug = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
    blob_name = f"{l1_prediction_name.lower()}_{timestamp_slug}_{uuid.uuid4().hex[:6]}.jpg"
    
    image_base_url = upload_image_to_blob(image_bytes, blob_name)
    
    #final processing time
    end_total_time = time.perf_counter()
    total_processing_time = end_total_time - start_total_time
    
    #final response
    metadata = {
        "classification_timestamp_utc": classification_timestamp,
        "total_processing_time_seconds": total_processing_time,
        "l1_inference_time_seconds": l1_inference_time,
        "l2_inference_time_seconds": l2_inference_time
    }
    return {"layer1_result": result_l1, "layer2_result": result_l2, "image_url": image_base_url, "metadata": metadata}

def run_health():
    """Logic for Health Check."""
    return {
        "status": "healthy",
        "models_loaded": bool(model_l1 and model_l2),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

# --- 8. The REQUIRED run() Function ---
def run(request):
    """
    Called for every single POST request.
    This function now routes based on a 'mode' key in the JSON body.
    """
    raw_data = request
    
    try:
        # 1. Parse the raw JSON data from the request
        try:
            data_json = json.loads(raw_data)
        except Exception as e:
            return {"error": f"Invalid JSON input: {str(e)}", "data": str(raw_data)}, 400

        # 2. Get the 'mode' from the JSON. Default to 'prod'.
        mode = data_json.get("mode", "prod")
        print(f"Request received. Routing to: '{mode}'")

        # 3. Route to the correct logic
        if mode == 'health':
            response_data = run_health()
            return response_data

        # --- All other modes need an image ---

        # 4. Validate the input using Pydantic
        try:
            data = ImageInput(**data_json)
        except ValidationError as e:
            return {"error": f"Invalid input schema: {str(e)}"}, 400
        except TypeError: # Catches if data_json is not a dict
            return {"error": "Invalid input schema. Expecting a JSON object."}, 400

        # 5. Route to the correct prediction logic
        if mode == 'layer1':
            response_data = run_layer1(data)
        elif mode == 'layer2':
            response_data = run_layer2(data)
        elif mode == 'prod':
            response_data = run_prod(data)
        else:
            return {"error": f"Unknown mode: {mode}. Valid modes are: prod, layer1, layer2, health"}, 400
            
        return response_data

    except Exception as e:
        error_message = f"Error in run(): {str(e)}"
        print(error_message)
        return {"error": error_message, "data": str(raw_data)}, 500