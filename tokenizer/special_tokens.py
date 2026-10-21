"""Special token definitions."""
PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"

SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]
SPECIAL_TOKEN_IDS = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
