import os
import torch
from torch_geometric.data import InMemoryDataset, Data


def sample_mask(idx, size):
    """
    Create a boolean mask of given size, with True at the specified indices.
    """
    mask = torch.zeros(size, dtype=torch.bool)
    mask[list(idx)] = True
    return mask


class MGTAB(InMemoryDataset):
    """
    MGTAB dataset loader for PyG InMemoryDataset.

    Expects raw PT files in <root>/raw/ and saves processed file in <root>/processed/.

    Required raw files under raw_dir:
      - edge_index.pt
      - edge_type.pt
      - edge_weight.pt
      - labels_stance.pt
      - labels_bot.pt
      - features.pt
    """

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        # Load the processed data with full (pickle) loading to support PyG Data objects
        processed_path = self.processed_paths[0]
        self.data, self.slices = torch.load(processed_path, weights_only=False)

    @property
    def raw_file_names(self):
        return [
            'edge_index.pt',
            'edge_type.pt',
            'edge_weight.pt',
            'labels_stance.pt',
            'labels_bot.pt',
            'features.pt',
        ]

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def raw_dir(self):
        # Directory for raw files: <root>/raw
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self):
        # Directory for processed files: <root>/processed
        return os.path.join(self.root, 'processed')

    def download(self):
        # No download; assume raw files are provided in raw_dir manually
        pass

    def process(self):
        """
        Read raw PT files from raw_dir, construct a PyG Data object,
        and save (data, slices) to processed_paths[0].
        """
        # Load edge information
        edge_index = torch.load(os.path.join(self.raw_dir, 'edge_index.pt'))
        # Avoid unnecessary copy warnings by using clone
        edge_index = edge_index.clone().detach().to(torch.long)

        edge_type = torch.load(os.path.join(self.raw_dir, 'edge_type.pt'))
        edge_weight = torch.load(os.path.join(self.raw_dir, 'edge_weight.pt'))

        # Load labels
        stance_label = torch.load(os.path.join(self.raw_dir, 'labels_stance.pt'))
        bot_label = torch.load(os.path.join(self.raw_dir, 'labels_bot.pt'))

        # Load features
        features = torch.load(os.path.join(self.raw_dir, 'features.pt'))
        features = features.to(torch.float)

        # Create Data object
        data = Data(x=features, edge_index=edge_index)
        data.edge_type = edge_type
        data.edge_weight = edge_weight
        data.y_stance = stance_label
        data.y_bot = bot_label

        # Create train/val/test masks
        num_samples = data.y_bot.size(0)
        train_end = int(0.7 * num_samples)
        val_end = int(0.9 * num_samples)

        data.train_mask = sample_mask(range(train_end), num_samples)
        data.val_mask = sample_mask(range(train_end, val_end), num_samples)
        data.test_mask = sample_mask(range(val_end, num_samples), num_samples)

        # Filter and transform
        data_list = [data]
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        # Save processed data
        data, slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), os.path.join(self.processed_dir, self.processed_file_names[0]))
