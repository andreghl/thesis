from torch.utils.data import Dataset
import h5py


class H5Dataset(Dataset):

    def __init__(self, path : str,
                 features : list[str],
                 target : str):
        """
        Args:
            path: the location of the .h5 file.
            features: the independent variables to load from the dataset.
            target: the dependent variable of the dataset.
        """
        self.path = path
        self.features = features
        self.target = target

        self.file = h5py.File(path, 'r')
        self.length = len(self.file[target])

        missing = [feature for feature in set(features + [target])
                   if feature not in self.file.keys()]

        if missing:
            message = f"""
            Features {missing} not found at {self.path}.\n 
            Available keys: {tuple(self.file.keys())}.
            """
            self.file.close()
            raise KeyError(message)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = {}
        target = self.file[self.target][idx]
        for feature in self.features:
            item[feature] = self.file[feature][idx]

        return *item.values(), target

    def __del__(self):
        if hasattr(self, 'file'):
            self.file.close()