"""
Download models from Cloudflare R2 (private bucket) to the local volume.
Runs once at container startup if models are not already present.

v6.0: 3 quantile models (Q25/Q50/Q75) + 1 meta-model.
"""

import os
import boto3
from pathlib import Path

MODELS_DIR = Path("/app/models")

FILES = [
    "model_1H_Q25.joblib",
    "model_1H_Q50.joblib",
    "model_1H_Q75.joblib",
    "meta_confidence.joblib",
]


def download():
    # Always re-download to ensure latest models from R2
    print(f"Downloading {len(FILES)} model(s) from R2...")

    account_id = os.environ["R2_ACCOUNT_ID"]
    access_key = os.environ["R2_ACCESS_KEY_ID"]
    secret_key = os.environ["R2_SECRET_ACCESS_KEY"]
    bucket = os.environ.get("R2_BUCKET", "lumeny-models")

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for fname in FILES:
        dest = MODELS_DIR / fname
        print(f"Downloading {fname}...")
        s3.download_file(bucket, fname, str(dest))
        print(f"  -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    print("All models downloaded.")


if __name__ == "__main__":
    download()
