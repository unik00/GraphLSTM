import en_core_sci_md

nlp = en_core_sci_md.load()
print("Finished loading en_core_sci_md")


def get_all_dep():
    for label in nlp.get_pipe("parser").labels:
        print(label)
    return nlp.get_pipe("parser").labels


def gen_dependency_tree(sent):
    doc = nlp(sent)
    return doc


def get_universal_POS():
    universal_pos = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]

    POS_map = dict()
    cnt = 0
    for s in universal_pos:
        POS_map[s] = cnt
        cnt += 1

    return POS_map


if __name__ == "__main__":
    s = "A bilateral retrobulbar neuropathy with an unusual central bitemporal hemianopic scotoma was found"
    print(get_universal_POS())