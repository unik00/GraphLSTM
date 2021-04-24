def get_universal_POS():
    universal_pos = ["ADJ",
                     "ADP",
                     "ADV",
                     "AUX",
                     "CONJ",
                     "DET",
                     "INTJ",
                     "NOUN",
                     "NUM",
                     "PART",
                     "PRON",
                     "PROPN",
                     "PUNCT",
                     "SCONJ",
                     "SYM",
                     "VERB",
                     "X"]
    POS_map = dict()
    cnt = 0
    for s in universal_pos:
        POS_map[s] = cnt
        cnt += 1

    return POS_map

