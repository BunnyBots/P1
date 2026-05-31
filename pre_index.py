import os
import json
import numpy as np
from PIL import Image
import face_recognition
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GALLERY_DIR = os.path.join(BASE_DIR, "gallery")
CACHE_PATH = os.path.join(GALLERY_DIR, ".pixify_cache.json")

def load_and_resize_image(image_path: str, max_dimension: int = 600) -> np.ndarray:
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

def pre_index():
    print(f"=== Pre-indexing Gallery Folder: {GALLERY_DIR} ===")
    if not os.path.exists(GALLERY_DIR):
        print(f"[ERROR] Gallery directory {GALLERY_DIR} does not exist.")
        return
        
    # Load existing cache
    cache = {}
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r") as f:
                cache = json.load(f)
            print(f"Loaded existing cache with {len(cache)} entries.")
        except Exception as e:
            print(f"Error loading existing cache: {e}")

    # Gather image files
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    all_files = os.listdir(GALLERY_DIR)
    image_names = [f for f in all_files if os.path.splitext(f)[1].lower() in valid_extensions]
    
    print(f"Found {len(image_names)} images to index.")
    
    cache_updated = False
    
    # Process images with progress bar
    for name in tqdm(image_names, desc="Indexing faces"):
        img_path = os.path.join(GALLERY_DIR, name)
        try:
            mtime = os.path.getmtime(img_path)
            
            # Skip if already cached and mtime matches
            if name in cache and cache[name].get("mtime") == mtime:
                continue
                
            image = load_and_resize_image(img_path, max_dimension=600)
            encodings = face_recognition.face_encodings(image)
            
            cache[name] = {
                "mtime": mtime,
                "encodings": [enc.tolist() for enc in encodings]
            }
            cache_updated = True
            
            # Save incrementally every 10 images to prevent loss of progress
            if cache_updated and len(cache) % 10 == 0:
                with open(CACHE_PATH, "w") as f:
                    json.dump(cache, f)
                    
        except Exception as e:
            print(f"\n[WARNING] Skipping file {name}: {e}")
            
    if cache_updated:
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f)
        print(f"\nCache successfully saved to {CACHE_PATH}")
        print(f"Total indexed images: {len(cache)}")
    else:
        print("\nAll images are already up-to-date in cache.")

if __name__ == "__main__":
    pre_index()
