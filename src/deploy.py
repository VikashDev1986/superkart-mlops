"""
deploy.py — Push deployment files to Hugging Face Space.
Environment variables required:
  HF_TOKEN     : Hugging Face write token
  HF_USERNAME  : Hugging Face username  (default: vikashHugFace)
"""
import os, warnings
from huggingface_hub import HfApi, create_repo

os.environ.setdefault("CURL_CA_BUNDLE",     "")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
import requests
_orig = requests.Session.request
def _no_verify(self, method, url, **kwargs):
    kwargs.setdefault("verify", False)
    return _orig(self, method, url, **kwargs)
requests.Session.request = _no_verify

HF_TOKEN      = os.environ["HF_TOKEN"]
HF_USERNAME   = os.environ.get("HF_USERNAME", "vikashHugFace")
HF_SPACE_REPO = f"{HF_USERNAME}/superkart-sales-forecast-space"
DEPLOY_DIR    = "deployment"

create_repo(HF_SPACE_REPO, repo_type="space", space_sdk="docker",
            exist_ok=True, token=HF_TOKEN)
print(f"Space: https://huggingface.co/spaces/{HF_SPACE_REPO}")

api = HfApi(token=HF_TOKEN)
for fname in ["Dockerfile", "requirements.txt", "app.py"]:
    api.upload_file(path_or_fileobj=f"{DEPLOY_DIR}/{fname}",
                    path_in_repo=fname, repo_id=HF_SPACE_REPO,
                    repo_type="space")
    print(f"Uploaded {fname}")
print("Deployment complete.")
