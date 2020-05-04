from torchtext.vocab import Vocab
from src.run import get_pad_to_min_len_fn


def test_postprocess():
    MIN_LENGTH = 10

    batch_sample = [
        list(range(10, 15)),
        list(range(5, 12)),
        list(range(12, 20))
    ]
    print(batch_sample)

    vocab_sample = Vocab({'<unk>': 0, '<pad>': 1, 'two': 2, 'three': 3, 'four': 4},
                         specials=['<unk>', '<pad>'])
    postproc = get_pad_to_min_len_fn(min_length=MIN_LENGTH)

    batch_pst = postproc(batch_sample, vocab_sample)

    min_length_batch = min([len(x) for x in batch_pst])
    print(min_length_batch)
    assert min_length_batch >= MIN_LENGTH
