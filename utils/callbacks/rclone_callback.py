from pytorch_lightning.callbacks import Callback
import subprocess

class RcloneUploadCallback(Callback):
        def __init__(self, local_dir, remote_dir):
            super().__init__()
            self.local_dir = local_dir  # Directory to save local checkpoints
            self.remote_dir = remote_dir  # OneDrive remote directory

        def on_epoch_end(self, trainer, pl_module):
            if trainer.current_epoch % trainer.check_val_every_n_epoch ==0:
                self._rclone_upload()
        
        def _rclone_upload(self):
            print("Training complete. Final upload to OneDrive...")
            # Upload one last time when training finishes
            command = f"rclone move --update --ignore-existing --no-traverse --verbose {self.local_dir} {self.remote_dir}"
            try:
                subprocess.run(command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error during upload: {e}")
            print("Final upload completed.")