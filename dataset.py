import collections 
import numpy as np
import os
def get_class(fname):
  if'NonViolence' in fname  :
    return 'NonViolence'
  else:
    return 'Violence'


def get_files_per_class(files):
  """ Retrieve the files that belong to each class.

    Args:
      files: List of files in the dataset.

    Returns:
      Dictionary of class names (key) and files (values).
  """
  files_for_class = collections.defaultdict(list)
  for fname in files:
    class_name = get_class(fname)
    files_for_class[class_name].append(fname)
  return files_for_class


def split_class_lists(files_for_class, count):
  """ Returns the list of files belonging to a subset of data as well as the remainder of
    files that need to be downloaded.

    Args:
      files_for_class: Files belonging to a particular class of data.
      count: Number of files to download.

    Returns:
      Files belonging to the subset of data and dictionary of the remainder of files that need to be downloaded.
  """
  split_files = []
  remainder = {}
  for cls in files_for_class:
    split_files.extend(files_for_class[cls][:count])
    remainder[cls] = files_for_class[cls][count:]
  return split_files, remainder

from pathlib import Path

def list_all_files_pathlib(directory_path):
    """
    Recursively lists all files in a given directory using pathlib.

    Args:
        directory_path (str): The path to the starting directory.

    Returns:
        list: A list of full file paths as Path objects.
    """
    # Create a Path object from the string path
    directory = Path(directory_path)

    # rglob('*') recursively finds all files and directories
    return [str(path) for path in directory.rglob('*') if path.is_file()]

print(len(list_all_files_pathlib('./Real Life Violence Dataset')))
import json
def get_dataset(path, train=0.7, test=0.2, val=0.1, cache_file='dataset_cache.json'):
  """
  Load datasets from cache if it exists, otherwise create and save it.
  
  Args:
    files: List of file paths
    train: Training split proportion
    test: Test split proportion
    val: Validation split proportion
    cache_file: Path to cache file
    
  Returns:
    Dictionary with train, val, test datasets
  """
  files = list_all_files_pathlib(path)
  # Check if cache exists
  if os.path.exists(cache_file):
    print('file from cache')
    with open(cache_file, 'r') as f:
      return json.load(f)
  
  # Group files by class
  files_for_class = {}
  for path in files:
    class_name = get_class(path)
    if class_name not in files_for_class:
      files_for_class[class_name] = []
    files_for_class[class_name].append(path)

  dataset = {
      'train': [],
      'test': [],
      'val': []
  }

  # For each class, split according to proportions and add to dataset
  for class_name, class_files in files_for_class.items():
    class_size = len(class_files)
    shuffled_indices = np.random.permutation(class_size)
    train_end = int(train * class_size)
    val_end = int((train + val) * class_size)

    class_files = np.array(class_files)[shuffled_indices]
    train_files = class_files[:train_end]
    val_files = class_files[train_end:val_end]
    test_files = class_files[val_end:]

    # Add (file, class) pairs
    dataset['train'].extend([(f, class_name) for f in train_files])
    dataset['val'].extend([(f, class_name) for f in val_files])
    dataset['test'].extend([(f, class_name) for f in test_files])
  
  # Save to cache
  with open(cache_file, 'w') as f:
    json.dump(dataset, f)
  
  return dataset

# dataset = get_dataset('./Real Life Violence Dataset')
# print('Train: ', len(dataset['train']))
# print('Val: ', len(dataset['val']))
# print('Test: ', len(dataset['test']))