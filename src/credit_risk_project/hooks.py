import subprocess
import logging
from kedro.framework.hooks import hook_impl

class CloudSyncHook:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.logger = logging.getLogger(__name__)

    @hook_impl
    def after_pipeline_run(self) -> None:
        """Synchronizes the local data directory to GCP storage."""
        self.logger.info(f"Synchronizing local data to gs://{self.bucket_name}...")
        try:
            # Use gsutil rsync to mirror local data/ to the bucket
            # -m: multithreaded
            # -r: recursive
            # -d: delete files in destination not in source (optional, excluded here for safety)
            subprocess.run(
                ["gsutil", "-m", "rsync", "-r", "data/", f"gs://{self.bucket_name}/"],
                check=True,
                capture_output=True,
                text=True
            )
            self.logger.info("GCP Synchronization complete.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"GCP Synchronization failed: {e.stderr}")
        except FileNotFoundError:
            self.logger.error("GCP Synchronization failed: 'gsutil' command not found. Ensure Google Cloud SDK is installed.")
