import torch
from torch.utils.data import Dataset


class LimitedDataset(Dataset):
    def __init__(self, sequence_file_path: str, boundary_file_path: str, max_lines=10000):
        self.sequence_file_path = sequence_file_path
        self.boundary_file_path = boundary_file_path
        self.max_lines = max_lines

    def __len__(self):
        # Assuming both files have the same number of lines
        with open(self.sequence_file_path, 'r') as file:
            for i, _ in enumerate(file):
                if i >= self.max_lines:
                    return self.max_lines
        return i

    def __getitem__(self, idx):
        if idx >= self.max_lines:
            raise IndexError("Index out of bounds")

        sequence = self.read_sequence_line(idx)
        boundary = self.read_boundary_line(idx)

        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(boundary, dtype=torch.float32)

    def read_sequence_line(self, idx: int) -> list[int]:
        with open(self.sequence_file_path, 'r') as file:
            for i, line in enumerate(file):
                if i == idx:
                    return self.parse_line(line)
        return []

    def read_data(self):
        line_count = 0
        start_reading = False
        with open(self.file_path, 'r') as file:
            for line in file:
                if line.strip() == '####END####':
                    start_reading = True
                    continue
                if start_reading:
                    if line_count < self.max_lines:
                        sequence: list[dict] = self.parse_line(line)
                        [self.data.append(i) for i in sequence]
                        line_count += 1
                    else:
                        break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx < len(self.data):
            return torch.tensor(self.data[idx], dtype=torch.float32)
        raise IndexError("Index out of bounds")

    def parse_line(self, line: str) -> list[list[int]]:
        line = line.strip()[1:-1]
        # Remove leading/trailing whitespace and split by '],'
        vectors = line.strip().split('],')
        # Process each vector
        sequence = []
        for vector in vectors:
            # Clean up the vector string and convert to integers
            cleaned_vector = vector.strip('[] \n').split(',')
            # Filter out empty strings and convert the rest to integers
            int_vector = [int(x) for x in cleaned_vector if x.strip()]
            sequence.append(int_vector)
        return sequence