import torch
from torch.utils.data import Dataset

MAXIMAL_BP_PER_BATCH = 100000


def read_boundary_line(sequence: list[int]) -> list[int]:
    """
    Determine if base pair contains gene_start, gene_end or none of these.
    :param sequence: Sequence of vectors. Each vector represents bp with features.
    :return: The boundary of gene.
    a) [1, 0, 0] == current bp has neither gene_start nor gene_end
    b) [0, 1, 0] == current bp has gene_start
    c) [0, 0, 1] == current bp has gene_end
    """
    gene_boundary: int = sequence[len(sequence) - 1]
    if gene_boundary == 0:
        return [1, 0, 0]
    elif gene_boundary == 1:
        return [0, 1, 0]
    return [0, 0, 1]


class LimitedDataset(Dataset):
    def __init__(self, sequence_file_path: str, bp_per_batch: int = 2, testing_data: bool = False):
        self.sequence_file_path = sequence_file_path
        self.bp_per_batch = bp_per_batch
        if self.bp_per_batch > MAXIMAL_BP_PER_BATCH:
            raise "Maximal BP limit reached"
        self.testing_data = testing_data
        self.data = []
        self.read_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        end_idx: int = min(idx + self.bp_per_batch, len(self.data))
        if idx >= len(self.data):
            raise IndexError("Index out of bounds")

        inputs_sequence: list[list[int]] = [self.data[i] for i in range(idx, end_idx)]
        targets_sequence: list[list[int]] = [read_boundary_line(i) for i in inputs_sequence]

        return torch.tensor(inputs_sequence, dtype=torch.float32), torch.tensor(targets_sequence, dtype=torch.float32)

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
                    sequences: list[list[int]] = self.parse_line(line)
                    [self.data.append(i) for i in sequences]
                    bp_count += len(sequences)

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

            if self.testing_data:
                int_vector = int_vector[:-1]  # Remove the last element if testing_data is True

            sequence.append(int_vector)
        return sequence
