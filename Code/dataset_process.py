from datasets import load_dataset, concatenate_datasets
import formatter as f


def extract_dataset():
    f.my_print("Extracting VoxPopuli-it dataset...")
    ds = load_dataset("facebook/voxpopuli", "it")
    f.my_print("Dataset correctly extracted!")

    train = ds['train']
    validation = ds['validation']
    test = ds['test']

    return concatenate_datasets([train, validation, test])

