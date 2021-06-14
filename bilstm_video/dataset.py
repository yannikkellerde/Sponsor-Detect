from torch.utils.data import Dataset

class SequenzDataset(Dataset):
    def __init__(self, input_sequenzes, target_sequenzes) -> None:
        super().__init__()
        self.input_sequenzes = input_sequenzes
        self.target_sequenzes = target_sequenzes

    def __getitem__(self, index):
        return (self.input_sequenzes[index], self.target_sequenzes[index])

    def __len__(self):
        return len(self.input_sequenzes)