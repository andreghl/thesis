import json
import os

def check_path(filepath: str):
    return os.path.isdir(filepath)

def save_params(params : dict,
                score: float | int,
                path: str = "data/logs/params/",
                model_name: str | None = None):

    if model_name is None:
        model_name = 'unnamed'

    os.makedirs(path, exist_ok = True)

    index = 0
    filename = f"{path}{model_name}_{index}.json"
    while check_path(filename):
        index += 1
        filename = f"{path}{model_name}_{index}.json"

    data = params
    data['score'] = score

    with open(filename, 'w') as f:
        json.dump(data, f)


def read_params(path: str = "data/logs/params/GainNN_0.json"):

    params = {}

    os.makedirs(path, exist_ok = True)
    if check_path(path):
        with open(path, 'w') as f:
            params = json.load(f)
    else:
        print(f"File {path} does not exist!")

    return params



