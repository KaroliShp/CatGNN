class SemiSupervisedDataset:
    def __init__(self):
        super(SemiSupervisedDataset, self).__init__()

    def split(self):
        raise NotImplementedError

    def get_split_masks(self):
        raise NotImplementedError

    def get_features(self):
        raise NotImplementedError

    def get_edges(self, sender_to_receiver=True):
        raise NotImplementedError

    def get_vertices(self):
        raise NotImplementedError

    def get_dimensions(self):
        raise NotImplementedError
