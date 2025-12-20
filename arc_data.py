"""
SETUP SCRIPT: Download ARC-AGI Dataset (Robust)
===============================================
Downloads the official Abstraction and Reasoning Corpus (ARC).
Auto-detects folder structure to prevent path errors.
"""

import os
import requests
import zipfile
import io
import shutil
import logging
from pathlib import Path

# Config
ARC_REPO_ZIP = "https://github.com/fchollet/ARC/archive/refs/heads/master.zip"
TARGET_DIR = "data"

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("Setup")

def setup_data():
    logger.info("📦 STARTING ARC DATA SETUP")
    
    final_path = os.path.join(TARGET_DIR, "training")
    
    # 1. Download
    logger.info(f"   ⬇️  Downloading from {ARC_REPO_ZIP}...")
    try:
        r = requests.get(ARC_REPO_ZIP)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        
        # 2. Find the correct internal folder
        # We look for any file that ends with "data/training/..."
        # This makes it robust against "ARC-master" vs "ARC-main" naming
        training_files = [f for f in z.namelist() if "data/training/" in f and f.endswith(".json")]
        
        if not training_files:
            raise ValueError("Could not find 'data/training' folder inside the zip.")
            
        logger.info(f"   🔍 Found {len(training_files)} task files.")
        
        # 3. Extract and Move
        os.makedirs(final_path, exist_ok=True)
        
        count = 0
        for file_in_zip in training_files:
            # Get just the filename (e.g., "007bbfb7.json")
            filename = os.path.basename(file_in_zip)
            if not filename: continue
            
            # Read content directly from zip
            with z.open(file_in_zip) as source, open(os.path.join(final_path, filename), "wb") as target:
                shutil.copyfileobj(source, target)
            count += 1
            
        logger.info(f"   ✅ Success! Extracted {count} tasks to '{final_path}'")
        
    except Exception as e:
        logger.error(f"   ❌ Download Failed: {e}")
        logger.error("   Alternative: Manually clone 'https://github.com/fchollet/ARC' and move 'data/training' folder here.")

if __name__ == "__main__":
    setup_data()