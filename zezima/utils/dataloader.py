import torch
from torch.utils.data import Dataset


class LimitedDataset(Dataset):
    def __init__(self, sequence_file_path: str, bp_per_batch: int=2, max_bp=10000):
        self.sequence_file_path = sequence_file_path
        self.max_bp = max_bp
        self.bp_per_batch = bp_per_batch
        self.data = []
        self.read_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        end_idx: int = min(idx + self.bp_per_batch, len(self.data))
        if idx >= len(self.data):
            raise IndexError("Index out of bounds")
        if end_idx > self.max_bp:
            raise IndexError("Index out of bounds")

        inputs_sequence: list[list[int]] = [self.data[i] for i in range(idx, end_idx)]
        targets_sequence: list[list[int]] = [self.read_boundary_line(i) for i in inputs_sequence]

        return torch.tensor(inputs_sequence, dtype=torch.float32), torch.tensor(targets_sequence, dtype=torch.float32)

    def read_boundary_line(self, sequence: list[int]):
        gene_boundary: int = sequence[len(sequence) -1]
        if gene_boundary == 0:
            return [1, 0, 0]
        elif gene_boundary == 1:
            return [0, 1, 0]
        return [0, 0, 1]

    def read_sequence_line(self, idx: int) -> list[list[int]]:
        with open(self.sequence_file_path, 'r') as file:
            for i, line in enumerate(file):
                if i == idx:
                    return self.parse_line(line)
        return []

    def read_data(self):
        bp_count = 0
        start_reading = False
        with open(self.sequence_file_path, 'r') as file:
            for line in file:
                if line.strip() == '####END####':
                    start_reading = True
                    continue
                if start_reading:
                    if bp_count < self.max_bp:
                        sequences: list[list[int]] = self.parse_line(line)
                        [self.data.append(i) for i in sequences]
                        bp_count += len(sequences)
                    else:
                        break

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