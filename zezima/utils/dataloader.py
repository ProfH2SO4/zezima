import ast
import math
import torch
import os
import multiprocessing

from multiprocessing import Pool, Manager
from torch.utils.data import Dataset

from zezima import log

MAXIMAL_BP_PER_BATCH = 100000


def read_boundary_line(
    sequence: list[int],
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
    last_pos: int = sequence.pop(len(sequence) -1)
    if last_pos == 0:
        return [0, 0, 0, 0]
    elif last_pos == 1:
        return [0, 1, 0, 0]
    elif last_pos == 2:
        return [0, 0, 1, 0]
    return [0, 0, 0, 1]


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


def estimate_line_count(file_path, sample_size=1024 * 1024):
    with open(file_path, "r") as file:
        sample = file.read(sample_size)
        estimated_lines = sample.count("\n")
        return os.path.getsize(file_path) // (len(sample) // estimated_lines)


def parse_line(line: str) -> list[list[int]]:
    return [process_vector(vector) for vector in line.strip("\n").split("\t")]


def process_vector(vector: str) -> list[int]:
    processed_vector = [int(element) for element in ast.literal_eval(vector)]
    return processed_vector


def process_file_section(args):
    file_path, start_line, end_line, shared_list, index = args
    local_data = []
    with open(file_path, "r", encoding="utf-8") as file:
        # Skip lines until the start_line
        for _ in range(start_line):
            next(file)
        # Process lines up to end_line
        for line_num, line in enumerate(file, start=start_line):
            if line_num > end_line:
                break
            line_s = line.strip()
            if not line_s.startswith("#"):
                r = parse_line(line_s)
                [local_data.append(i) for i in r]
    shared_list[index] = local_data


def split_file_processing(file_path, num_cores):
    total_lines = estimate_line_count(file_path)
    lines_per_process = total_lines // num_cores

    with Manager() as manager:
        shared_list = manager.list(
            [[] for _ in range(num_cores)]
        )  # Initialize shared list

        # Create arguments for each process, ensuring each process starts at the beginning of a line
        args = [
            (
                file_path,
                i * lines_per_process,
                (total_lines if i == num_cores - 1 else (i + 1) * lines_per_process)
                - 1,
                shared_list,
                i,
            )
            for i in range(num_cores)
        ]

        with Pool(processes=num_cores) as pool:
            pool.map(process_file_section, args)

        combined_result = [item for sublist in shared_list for item in sublist]

    return combined_result


class LimitedDataset(Dataset):
    def __init__(
        self,
        sequence_file_path: str,
        d_model: int,
        cpu_cores: int,
        bp_per_batch: int = 2,
        testing_data: bool = False,
        feature_to_find: str = "gene",
    ):
        self.sequence_file_path = sequence_file_path
        self.bp_per_batch = bp_per_batch
        self.d_model = d_model
        if self.bp_per_batch > MAXIMAL_BP_PER_BATCH:
            raise "Maximal BP limit reached"
        if cpu_cores < 0 or cpu_cores > multiprocessing.cpu_count():
            raise "Problem with CPU cores"
        self.cpu_cores = cpu_cores
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
        targets_sequence: list[list[int]] = [read_boundary_line(j) for j in inputs_sequence]
        self.st_window = self.ed_window
        self.ed_window += self.bp_per_batch
        return torch.tensor(inputs_sequence, dtype=torch.float64), torch.tensor(
            targets_sequence, dtype=torch.float64
        )

    def read_data(self) -> None:
        log.info(f"Loading data from file using cpu_cores: {self.cpu_cores}")
        self.read_header()
        self.data = split_file_processing(
            self.sequence_file_path,
            self.cpu_cores,
        )

    def encode_data(self):
        from torch import nn
        feature_embedding_dim = 8
        # Define embedding layers for each categorical feature
        embedding_layers = {
            feature: nn.Embedding(num_embeddings=4, embedding_dim=feature_embedding_dim)
            # Assuming each feature can have up to 4 states for simplicity
            for feature in self.bp_vector_schema if feature not in ['A', 'C', 'G', 'T']  # Exclude binary features
        }

        for vector in self.data:
            # Process binary features (nucleotides) directly and add a new dimension at the end
            nucleotides = torch.tensor(vector[:4], dtype=torch.long).unsqueeze(
                0)  # Adding a dimension to match feature vectors

            # Initialize a list to hold the embeddings for categorical features
            feature_vectors = []

            # Start index for categorical features in the vector
            cat_feature_start_index = 4

            # Iterate over categorical features and their corresponding embedding layers
            for feature in self.bp_vector_schema[cat_feature_start_index:]:
                # Get the state for the current feature
                feature_state = torch.tensor([vector[cat_feature_start_index]]).long()

                # Get the embedding vector for the current feature state and remove the singleton dimension
                feature_vector = embedding_layers[feature](feature_state)  # Squeezing to match nucleotides dimension

                # Append the embedding vector to the list
                feature_vectors.append(feature_vector)

                # Increment the start index for the next feature
                cat_feature_start_index += 1

            # Concatenate all vectors (nucleotides + categorical feature embeddings) to form the final input vector for the model
            model_input = torch.cat([nucleotides] + feature_vectors, dim=-1)
            self.encoded_data.append(model_input)

    def read_header(self):
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

    def pad_data(self) -> None:
        if self.bp_per_batch > len(self.data):
            padding_needed = self.bp_per_batch - len(self.data)
        else:
            padding_needed = (
                self.bp_per_batch - (len(self.data) % self.bp_per_batch)
            ) % self.bp_per_batch

        if padding_needed > 0:
            last_element = self.data[-1] if self.data else None
            for _ in range(padding_needed):
                self.data.append(list(last_element))

    def reset_window(self) -> None:
        self.st_window = 0
        self.ed_window = self.bp_per_batch
