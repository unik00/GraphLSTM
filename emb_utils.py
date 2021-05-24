import en_core_sci_md


nlp = en_core_sci_md.load()
print("Finished loading en_core_sci_md")


def gen_dependency_tree(sent):
    doc = nlp(sent)
    return doc


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


if __name__ == "__main__":
    s = "A bilateral retrobulbar neuropathy with an unusual central bitemporal hemianopic scotoma was found"
    print(get_POSs(s))
