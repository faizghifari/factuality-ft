import json
import random
import requests

from urllib.parse import quote


def check_wikipedia_page(page_name):
    try:
        # Encode the page name for use in a URL
        encoded_page_name = quote(page_name)

        # Construct the Wikipedia URL
        url = f"https://en.wikipedia.org/wiki/{encoded_page_name}"

        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the response status code is 200 (OK)
        if response.status_code == 200:
            print(f"The Wikipedia page '{page_name}' exists.")
        else:
            raise ValueError(f"The Wikipedia page '{page_name}' does not exist.")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


def read_file_lines(filename):
    try:
        with open(filename, "r") as file:
            lines = file.readlines()
            # Stripping newline characters from the end of each line
            lines = [line.strip() for line in lines]
            return lines
    except FileNotFoundError:
        print("File not found.")
        return []


def write_list_to_file(file_path, text_list):
    with open(file_path, "w", encoding="utf-8") as file:
        for item in text_list:
            file.write(str(item) + "\n")


def take_random_elements(input_list, num_elements):
    if num_elements > len(input_list):
        print("Error: Number of elements to take exceeds the length of the list.")
        return []
    else:
        random_elements = random.sample(input_list, num_elements)
        return random_elements


def write_jsonl(list_of_dicts, filename):
    try:
        # Open a file for writing
        with open(filename, "w") as f:
            # Iterate over the list of dictionaries
            for item in list_of_dicts:
                # Convert each dictionary to a JSON string and write it to the file
                json_string = json.dumps(item)
                f.write(json_string + "\n")
        print(f"Data written to {filename} successfully.")
    except Exception as e:
        print(f"Error occurred while writing to {filename}: {e}")


def append_dict_to_jsonl(file_path, dictionary):
    """
    Appends a dictionary as a new JSON object to an existing .jsonl file.
    If the file doesn't exist, it creates a new file.

    Args:
        dictionary (dict): The dictionary to be appended as a JSON object.
        file_path (str): The path to the .jsonl file.
    """
    try:
        with open(file_path, "a", encoding="utf-8") as file:
            json_object = json.dumps(dictionary)
            file.write(json_object + "\n")
    except Exception as e:
        print(f"An error occurred: {e}")


def read_jsonl_file(file_path):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data_dict = json.loads(line)
            data_dict.pop("input")
            data_dict.pop("output")
            data_dict.pop("annotations")
            data_list.append(data_dict)
    return data_list


def batch_iterator(iterable, batch_size=10):
    """
    Iterate through an iterable in batches of a specified size.

    Args:
        iterable (list, tuple, or any iterable): The iterable to be batched.
        batch_size (int, optional): The size of each batch. Defaults to 10.

    Yields:
        list: A batch of items from the iterable.
    """
    iterator = iter(iterable)
    batch = []

    try:
        while True:
            for _ in range(batch_size):
                batch.append(next(iterator))
            yield batch
            batch = []
    except StopIteration:
        if batch:
            yield batch
