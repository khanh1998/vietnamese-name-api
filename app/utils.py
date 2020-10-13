from app.prepare_data import segment_word, encode_bpe, pad_bpe_code, create_mask
import torch
import numpy as np


def predict(X, model):
    """input: a list of names"""
    s = segment_word(X)
    e = encode_bpe(s)
    p = pad_bpe_code(e)
    t = torch.tensor(p)
    m = create_mask(t.numpy())
    m = torch.tensor(m)
    with torch.no_grad():
        res = model(input_ids=t, token_type_ids=None, attention_mask=m)
        return np.argmax(res[0].numpy(), axis=1) # res is a tuple