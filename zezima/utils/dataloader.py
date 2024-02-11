import ast
import math
import torch
from torch.utils.data import Dataset

MAXIMAL_BP_PER_BATCH = 100000


def read_boundary_line(
    sequence: list[int], positions_to_look: tuple[int, int]
) -> list[int]:
    """
    Determine if the specified boundary in the sequence contains gene_start, gene_end or none of these.
    :param sequence: Sequence of vectors. Each vector represents bp with features.
    :param positions_to_look: Tuple indicating the boundary positions in the sequence to look for gene features.
    :return: The boundary status of the gene.
    [1, 0, 0, 0] = no gene
    [0, 1, 0, 0] = gene_start
    [0, 0, 1, 0] = ongoing/middle
    [0, 0, 0, 1] = gene_end
    """
    start_pos, end_pos = positions_to_look
    boundary_status = sequence[start_pos:end_pos]
    boundary_status = [0] + boundary_status
    if sum(boundary_status) == 0:
        boundary_status[0] = 1
    return boundary_status


def calculate_feature_boundary(
    vector_schema: list[str], feature_to_find: str, max_feature_overlap: int
) -> tuple[int, int]:
    base_pairs_count = 4  # Assuming 'A', 'C', 'G', 'T' are always the first 4

    # Find the index of the feature in the vector schema
    feature_index = vector_schema.index(feature_to_find)

    # Calculate the start index of the feature in the expanded vector
    # Each non-base-pair feature is represented by 3 elements ([start, ongoing, end]) times max_feature_overlap
    start_index = (
        base_pairs_count + (feature_index - base_pairs_count) * 3 * max_feature_overlap
    )

    # Calculate the end index (exclusive) of the feature in the expanded vector
    end_index = start_index + 3 * max_feature_overlap

    return start_index, end_index


class LimitedDataset(Dataset):
    def __init__(
        self,
        sequence_file_path: str,
        d_model: int,
        bp_per_batch: int = 2,
        testing_data: bool = False,
        feature_to_find: str = "gene",
    ):
        self.sequence_file_path = sequence_file_path
        self.bp_per_batch = bp_per_batch
        self.d_model = d_model
        if self.bp_per_batch > MAXIMAL_BP_PER_BATCH:
            raise "Maximal BP limit reached"
        self.testing_data = testing_data
        self.feature_to_find = feature_to_find
        self.positions_to_look: tuple[int, int] = 0, 0
        self.max_feature_overlap = 1
        self.bp_vector_schema = []
        self.data = []
        self.read_data()
        self.pad_data()
        self.st_window = 0
        self.ed_window = self.bp_per_batch

    def __len__(self):
        return math.ceil(len(self.data) / self.bp_per_batch)

    def __getitem__(self, idx: int):

        if self.ed_window > len(self.data):
            raise IndexError("Index out of bounds")

        inputs_sequence: list[list[int]] = [self.data[i] for i in range(self.st_window, self.ed_window)]
        targets_sequence: list[list[int]] = [
            read_boundary_line(i, self.positions_to_look) for i in inputs_sequence
        ]
        self.st_window = self.ed_window
        self.ed_window += self.bp_per_batch
        return torch.tensor(inputs_sequence, dtype=torch.float64), torch.tensor(
            targets_sequence, dtype=torch.float64
        )

    def convert_vector_to_torch(self, vector):
        # Process each element in the vector. If an element is a list, convert it to a tensor
        return [
            torch.tensor(sub_vector, dtype=torch.float64)
            if isinstance(sub_vector, list)
            else sub_vector
            for sub_vector in vector
        ]

    def read_sequence_line(self, idx: int) -> list[list[int]]:
        with open(self.sequence_file_path, "r") as file:
            for i, line in enumerate(file):
                if i == idx:
                    return self.parse_line(line)
        return []

    def read_data(self):
        start_reading = False
        with open(self.sequence_file_path, "r") as file:
            for line in file:
                if "#bp_vector_schema" in line.strip():
                    self.bp_vector_schema = ast.literal_eval(line.split("=")[1].strip())
                if "#max_feature_overlap" in line.strip():
                    self.max_feature_overlap = int(line.split("=")[1].strip())
                if line.strip() == "####END####":
                    self.positions_to_look = calculate_feature_boundary(
                        self.bp_vector_schema,
                        self.feature_to_find,
                        self.max_feature_overlap,
                    )
                    start_reading = True
                    continue
                if line == "\n":
                    continue
                if start_reading:
                    sequences: list[list[int]] = self.parse_line(line)
                    [self.data.append(i) for i in sequences]

    def parse_line(self, line: str) -> list[list[int]]:
        return [self.process_vector(vector) for vector in line.strip("\n").split("\t")]

    def process_vector(self, vector: str) -> list[int]:
        processed_vector = [int(element) for element in ast.literal_eval(vector)]

        # Check if padding is needed
        if len(processed_vector) < self.d_model:
            # Calculate the number of padding elements needed
            padding_size = self.d_model - len(processed_vector)

            # Pad the vector with zeros
            processed_vector.extend([0] * padding_size)
        return processed_vector

    def pad_data(self) -> None:
        reminder: int = len(self.data) % self.bp_per_batch
        for i in range(reminder):
            self.data.append(self.data[len(self.data) - 1])

    def reset_window(self) -> None:
        self.st_window = 0
        self.ed_window = self.bp_per_batch