"""
    File reader, pre-processing, etc.
    Author: Pablo Botton da Costa

"""
from torchtext.datasets import TranslationDataset
from torchtext import data, datasets


def read_data(dir_path: str, source: str, target: str, batch_size: int, target_language: str, source_language: str):

    """
        Read data, tokenize it and batch it.
    """

    SRC = data.Field(tokenize="spacy",
                     tokenizer_language=source_language,
                     init_token='<sos>',
                     eos_token='<eos>',
                     pad_token='<pad>',
                     lower=True)

    TRG = data.Field(tokenize="spacy",
                     tokenizer_language=target_language,
                     init_token='<sos>',
                     eos_token='<eos>',
                     pad_token='<pad>',
                     lower=True)

    data_train = datasets.TranslationDataset(
        path='./data/data.train', exts=('.src', '.tgt'),
        fields=(SRC, TRG))
    data_test = datasets.TranslationDataset(
        path='./data/data.test', exts=('.src', '.tgt'),
        fields=(SRC, TRG))
    data_dev = datasets.TranslationDataset(
        path='./data/data.dev', exts=('.src', '.tgt'),
        fields=(SRC, TRG))

    SRC.build_vocab([data_train, data_dev], min_freq=2)
    TRG.build_vocab([data_train, data_dev], min_freq=2)

    return data.BucketIterator.splits(
        (data_train, data_dev, data_test),
        batch_size=batch_size,
        device="cuda")
