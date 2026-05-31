import os
import re
import base64
import shutil
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import face_recognition
import gdown

import json

def load_encodings_cache(directory: str) -> dict:
    cache_path = os.path.join(directory, ".pixify_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
    return {}

def save_encodings_cache(directory: str, cache: dict):
    cache_path = os.path.join(directory, ".pixify_cache.json")
    try:
        with open(cache_path, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"Error saving cache: {e}")

app = Flask(__name__)
# Enable CORS for all routes to support requests from Astro dev server (usually localhost:4321)
CORS(app)

# Configuration for temporary and download storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp")
DOWNLOADS_DIR = os.path.join(TEMP_DIR, "downloads")

# Ensure directories exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

def extract_folder_id(url: str) -> str:
    """
    Extracts the Google Drive folder ID or file ID from various URL formats.
    """
    # 1. Folders link (standard and sharing)
    folder_match = re.search(r'folders/([a-zA-Z0-9_-]+)', url)
    if folder_match:
        return folder_match.group(1)
        
    # 2. File link (fallback/sharing)
    file_match = re.search(r'file/d/([a-zA-Z0-9_-]+)', url)
    if file_match:
        return file_match.group(1)
        
    # 3. ID parameter
    id_match = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', url)
    if id_match:
        return id_match.group(1)
        
    # 4. Raw ID or Local Testing Keywords
    clean_url = url.strip()
    if clean_url in ("local_gallery", "gallery", "test_gallery") or re.match(r'^[a-zA-Z0-9_-]{28,45}$', clean_url):
        return clean_url
        
    raise ValueError("Invalid Google Drive URL format. Could not extract Folder/File ID.")

def load_and_resize_image(image_path: str, max_dimension: int = 1000) -> np.ndarray:
    """
    Loads an image and resizes it to speed up face recognition.
    """
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        width, height = img.size
        if width > max_dimension or height > max_dimension:
            try:
                resample_filter = Image.Resampling.LANCZOS
            except AttributeError:
                resample_filter = Image.LANCZOS
            img.thumbnail((max_dimension, max_dimension), resample_filter)
        return np.array(img)

def extract_anchor_embedding(image_path: str) -> np.ndarray:
    """
    Extracts the 128-dimensional face encoding vector of the primary face.
    """
    image = load_and_resize_image(image_path, max_dimension=1000)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        raise ValueError("No face detected in the reference image. Please use a clear portrait.")
    return np.array(encodings[0])

def create_thumbnail_base64(image_path: str, max_size: int = 300) -> str:
    """
    Loads an image, downscales it to thumbnail size, and returns a base64 JPEG string.
    """
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail((max_size, max_size))
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=75)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"Error creating thumbnail for {image_path}: {e}")
        return ""

def get_file_size_str(file_path: str) -> str:
    try:
        size_bytes = os.path.getsize(file_path)
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
    except Exception:
        return "Unknown Size"

def get_image_dimensions_str(image_path: str) -> str:
    try:
        with Image.open(image_path) as img:
            return f"{img.width}x{img.height}px"
    except Exception:
        return "Unknown dimensions"

@app.route('/')
def index():
    return jsonify({
        "status": "success",
        "message": "Pixify AI ML Model Server is running successfully!",
        "endpoints": {
            "/api/filter": "POST - Filters images using facial recognition"
        }
    })

@app.route('/api/filter', methods=['POST'])
def filter_images():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Missing JSON request body"}), 400
            
        google_drive_link = data.get("google_drive_link")
        reference_image_base64 = data.get("reference_image")
        similarity_threshold = float(data.get("similarity_threshold", 0.75))
        max_results = int(data.get("max_results", 50))
        
        if not google_drive_link:
            return jsonify({"status": "error", "message": "Missing parameter 'google_drive_link'"}), 400
        if not reference_image_base64:
            return jsonify({"status": "error", "message": "Missing parameter 'reference_image'"}), 400
            
        # 1. Parse Google Drive folder link
        try:
            folder_id = extract_folder_id(google_drive_link)
        except ValueError as ve:
            return jsonify({"status": "error", "message": str(ve)}), 400
            
        # 2. Decode reference image and save to temporary file
        if "," in reference_image_base64:
            header, base64_data = reference_image_base64.split(",", 1)
        else:
            base64_data = reference_image_base64
            
        ref_image_path = os.path.join(TEMP_DIR, f"ref_{folder_id}.jpg")
        try:
            with open(ref_image_path, "wb") as f:
                f.write(base64.b64decode(base64_data))
        except Exception as e:
            return jsonify({"status": "error", "message": f"Failed to decode reference image: {e}"}), 400
            
        # 3. Extract Anchor Embedding vector
        try:
            anchor_embedding = extract_anchor_embedding(ref_image_path)
        except ValueError as ve:
            # Clean up ref image first
            if os.path.exists(ref_image_path):
                os.remove(ref_image_path)
            return jsonify({"status": "error", "message": str(ve)}), 400
        except Exception as e:
            if os.path.exists(ref_image_path):
                os.remove(ref_image_path)
            return jsonify({"status": "error", "message": f"Biometric model error: {e}"}), 500
            
        # 4. Download Google Drive folder contents (skip if cached)
        if folder_id == "test_gallery":
            folder_download_path = os.path.join(TEMP_DIR, "test_gallery")
            os.makedirs(folder_download_path, exist_ok=True)
            files_in_test = os.listdir(folder_download_path)
            if not files_in_test or len(files_in_test) < 2:
                src_gallery = os.path.join(BASE_DIR, "gallery")
                if os.path.exists(src_gallery):
                    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
                    copied = 0
                    for f in os.listdir(src_gallery):
                        if os.path.splitext(f)[1].lower() in valid_exts and f != "reference_face.jpg":
                            shutil.copy(os.path.join(src_gallery, f), os.path.join(folder_download_path, f))
                            copied += 1
                            if copied >= 5:
                                break
                    ref_src = os.path.join(BASE_DIR, "reference_face.jpg")
                    if os.path.exists(ref_src):
                        shutil.copy(ref_src, os.path.join(folder_download_path, "reference_face.jpg"))
            is_cached = True
            print("Using small test gallery for quick offline verification.")
        elif folder_id in ("local_gallery", "gallery"):
            folder_download_path = os.path.join(BASE_DIR, "gallery")
            is_cached = True
            print("Using local workspace gallery folder for testing.")
        else:
            folder_download_path = os.path.join(DOWNLOADS_DIR, folder_id)
            
            # Check if cache folder exists and contains files
            is_cached = False
            if os.path.exists(folder_download_path):
                files_in_cache = [f for root, dirs, files in os.walk(folder_download_path) for f in files]
                if len(files_in_cache) > 0:
                    is_cached = True
                    print(f"Using cached downloads for folder: {folder_id} ({len(files_in_cache)} files found)")
                    
        if not is_cached:
            os.makedirs(folder_download_path, exist_ok=True)
            print(f"Downloading files from Google Drive folder: {folder_id}")
            try:
                gdown.download_folder(
                    id=folder_id,
                    output=folder_download_path,
                    quiet=False,
                    use_cookies=False
                )
            except Exception as e:
                # Clean up ref image
                if os.path.exists(ref_image_path):
                    os.remove(ref_image_path)
                return jsonify({
                    "status": "error", 
                    "message": f"Failed to download Google Drive folder contents. Ensure the folder is public ('Anyone with link can view'): {e}"
                }), 400
                
        # 5. Scan downloaded gallery folder recursively
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        image_paths = []
        for root, dirs, files in os.walk(folder_download_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    image_paths.append(os.path.join(root, file))
                    
        if not image_paths:
            # Clean up ref image
            if os.path.exists(ref_image_path):
                os.remove(ref_image_path)
            return jsonify({
                "status": "error", 
                "message": "No valid images found in the downloaded Google Drive folder. Supported formats: JPG, PNG, WebP."
            }), 400
            
        results = []
        cache = load_encodings_cache(folder_download_path)
        cache_updated = False
        
        # 6. Run facial recognition pipeline on gallery images
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            rel_path = os.path.relpath(img_path, folder_download_path)
            try:
                mtime = os.path.getmtime(img_path)
                cached_item = cache.get(rel_path)
                
                if cached_item and cached_item.get("mtime") == mtime:
                    encodings = [np.array(enc) for enc in cached_item.get("encodings", [])]
                else:
                    # Load and resize to 600px for 3x faster detection on CPU
                    image = load_and_resize_image(img_path, max_dimension=600)
                    encodings = face_recognition.face_encodings(image)
                    cache[rel_path] = {
                        "mtime": mtime,
                        "encodings": [enc.tolist() for enc in encodings]
                    }
                    cache_updated = True
                
                if not encodings:
                    continue
                    
                # Compute Euclidean distances
                distances = face_recognition.face_distance(encodings, anchor_embedding)
                min_distance = float(np.min(distances))
                
                # Convert to fraction similarity: 0.0 distance is 1.0 similarity
                similarity = max(0.0, 1.0 - min_distance)
                
                # Filter against the threshold
                if similarity >= similarity_threshold:
                    thumbnail_base64 = create_thumbnail_base64(img_path, max_size=300)
                    dimensions = get_image_dimensions_str(img_path)
                    size = get_file_size_str(img_path)
                    
                    results.append({
                        "filename": filename,
                        "similarity": round(similarity, 4),
                        "google_drive_url": google_drive_link,
                        "mock_thumbnail": thumbnail_base64,
                        "dimensions": dimensions,
                        "size": size
                    })
            except Exception as e:
                print(f"Skipping file {filename} due to processing error: {e}")
                
        if cache_updated:
            save_encodings_cache(folder_download_path, cache)
                
        # 7. Sort results in descending order of similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Limit to max results
        results = results[:max_results]
        
        # Clean up temporary reference image
        if os.path.exists(ref_image_path):
            os.remove(ref_image_path)
            
        print(f"Pipeline complete. Found {len(results)} matches.")
        return jsonify({
            "status": "success",
            "results": results
        })
        
    except Exception as e:
        print(f"Uncaught pipeline exception: {e}")
        return jsonify({"status": "error", "message": f"Internal server error: {e}"}), 500

if __name__ == '__main__':
    # Start Flask server locally on port 5000
    print("Starting Pixify AI ML Model Server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
