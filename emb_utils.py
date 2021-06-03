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
    # s = "A bilateral retrobulbar neuropathy with an unusual central bitemporal hemianopic scotoma was found"
    # print(get_universal_POS())
    s =  "Two cases of hepatic adenoma and one of focal nodular hyperplasia presumably associated with the use of oral contraceptives, are reported. Special reference is made to their clinical presentation, which may be totally asymptomatic. Liver-function tests are of little diagnostic value, but valuable information may be obtained from both liver scanning and hepatic angiography. Histologic differences and clinical similarities between hepatic adenoma and focal nodular hyperplasia of the liver are discussed."
    # from spacy import displacy
    doc = gen_dependency_tree(s)
    # displacy.serve(doc, style="dep")
    for x in doc:
        print(x)
