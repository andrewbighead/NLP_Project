from datasets import load_dataset, concatenate_datasets


def extract_dataset():
    ds = load_dataset("facebook/voxpopuli", "it")
    train = ds['train']
    validation = ds['validation']
    test = ds['test']
    return concatenate_datasets([train, validation, test])

