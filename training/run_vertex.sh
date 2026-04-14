#!/usr/bin/env bash
# Vertex AI entrypoint: download labels from GCS, train, upload results
set -euo pipefail

STRATEGY="${1:-A}"
BUCKET="gs://gotm-director-ai-training"

echo "=== Distillation v5 — Strategy: ${STRATEGY} ==="
echo "Downloading soft labels from GCS..."

mkdir -p training/output
# Use Application Default Credentials (Vertex AI provides them)
python3 -c "
from google.cloud import storage
client = storage.Client()
bucket = client.bucket('gotm-director-ai-training')
blob = bucket.blob('labels/distil_labels_v3_flipped.json')
blob.download_to_filename('training/output/distil_labels_v3_flipped.json')
print(f'Downloaded {blob.size} bytes')
"

echo "Starting training..."
python3 training/distil_v5_cloud.py --strategy "${STRATEGY}"

echo "Uploading results to GCS..."
python3 -c "
import glob, os
from google.cloud import storage
client = storage.Client()
bucket = client.bucket('gotm-director-ai-training')
for f in glob.glob('training/output/v5/**/*', recursive=True):
    if os.path.isfile(f):
        blob_name = f.replace('training/output/', 'results/')
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(f)
        print(f'Uploaded {f} → gs://gotm-director-ai-training/{blob_name}')
print('Done.')
"
