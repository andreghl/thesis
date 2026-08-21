from torch.utils.tensorboard import SummaryWriter
import os

def check_path(filepath: str):
    return os.path.isdir(filepath)


class Logger:

    def __init__(self,
                 log_dir: str,
                 label: str):
        os.makedirs(log_dir, exist_ok = True)

        index = 0
        path = f"{log_dir}/{label}_{index}"
        while check_path(path):
            index += 1
            path = f"{log_dir}/{label}_{index}"

        os.makedirs(path, exist_ok = True)
        self.writer = SummaryWriter(log_dir = path)

        print(f"Logging {label} at {path}")

    def log(self,
            metrics: dict["str", float | int],
            epoch: int):

        for key, value in metrics.items():
            self.writer.add_scalar(tag = key,
                                   scalar_value = value,
                                   global_step = epoch)
        self.writer.flush()

    def close(self):
        self.writer.close()