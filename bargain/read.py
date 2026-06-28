import h5py


def read_h5(path : str, datasets : list['str']):
    """Read a .h5 file and return prespecified datasets.

    Args:
        path: a string indicating the location of the .h5 file.
        datasets: a list of string containing the names of the datasets to load.

    Returns:
        A dictionary of the loaded datasets if they exist in the .h5 file.
    """
    data = {}

    with h5py.File(path, "r") as file:
        for name in datasets:
            if name not in file.keys():
                print(f"Warning! Name '{name}' not in {tuple(file.keys())} at {path}.")
                continue
            # get dataset and load into memory using [()]
            data[name] = file[name] [()]


    return data