import en_core_sci_md
from spacy import displacy


def load_en_core_sci_md():
    nlp = en_core_sci_md.load()
    print("Finished loading")
    return nlp


def gen_dependency_tree(nlp_model, sent):
    doc = nlp_model(sent)
    return doc


if __name__ == "__main__":
    nlp = load_en_core_sci_md()
    doc = gen_dependency_tree(nlp, "A bilateral retrobulbar neuropathy with an unusual central bitemporal hemianopic scotoma was found")
    for sent in doc:
        print(sent.pos_)

