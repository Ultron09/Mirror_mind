import os
import requests
import logging

# Setup
DATA_DIR = "data/evaluation"
REPO_URL = "https://api.github.com/repos/fchollet/ARC-AGI/contents/data/evaluation"

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("Downloader")

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    logger.info(f"⬇️ Fetching file list from {REPO_URL}...")
    try:
        response = requests.get(REPO_URL)
        response.raise_for_status()
        files = response.json()
        
        count = 0
        for file_info in files:
            if file_info['name'].endswith('.json'):
                raw_url = file_info['download_url']
                dest_path = os.path.join(DATA_DIR, file_info['name'])
                
                # Skip if exists
                if os.path.exists(dest_path):
                    continue
                    
                # Download
                r = requests.get(raw_url)
                with open(dest_path, 'wb') as f:
                    f.write(r.content)
                count += 1
                if count % 50 == 0: logger.info(f"   Downloaded {count} tasks...")
                
        logger.info(f"✅ Success: {count} new tasks downloaded to {DATA_DIR}/")
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")

if __name__ == "__main__":
    download_data()
    