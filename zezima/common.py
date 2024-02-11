import os
def create_file_if_not_exists(path_to_file: str) -> None:
    # Check if the directory exists, and create it if it doesn't
    directory = os.path.dirname(path_to_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Check if the file exists, and create it if it doesn't
    if not os.path.exists(path_to_file):
        with open(path_to_file, "w") as file:
            pass