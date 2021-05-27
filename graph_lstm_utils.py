import numpy as np

from emb_utils import gen_dependency_tree


class AdjMatrixBuilder:

    SPACY_DEP_LIST = ["ROOT",
                "acl",
                "acl:relcl",
                "acomp",
                "advcl",
                "advmod",
                "amod",
                "amod@nmod",
                "appos",
                "attr",
                "aux",
                "auxpass",
                "case",
                "cc",
                "cc:preconj",
                "ccomp",
                "compound",
                "compound:prt",
                "conj",
                "cop",
                "csubj",
                "dative",
                "dep",
                "det",
                "det:predet",
                "dobj",
                "expl",
                "intj",
                "mark",
                "meta",
                "mwe",
                "neg",
                "nmod",
                "nmod:npmod",
                "nmod:poss",
                "nmod:tmod",
                "nsubj",
                "nsubjpass",
                "nummod",
                "parataxis",
                "pcomp",
                "pobj",
                "preconj",
                "predet",
                "prep",
                "punct",
                "quantmod",
                "xcomp"]

    def __init__(self):
        self.DEP_MAP = dict()

        self.num_edge_type = 0
        for dep in self.SPACY_DEP_LIST:
            self.DEP_MAP[dep] = self.num_edge_type
            self.num_edge_type += 1

        self.SELF_ARC = self.num_edge_type
        self.num_edge_type += 1
        self.NEXT_ARC = self.num_edge_type
        self.num_edge_type += 1

    def __call__(self, doc):
        """
        Args:
            doc: spacy doc

        Returns:
            2D numpy array denoting adjacency matrix
        """
        matrix = np.zeros(dtype=np.int16, shape=(len(doc), len(doc)))

        for i in range(len(doc)):
            # self arc
            matrix[i][i] = self.SELF_ARC

            # next arc
            if i < len(doc) - 1:
                matrix[i][i + 1] = self.NEXT_ARC

            # out dependency arcs
            # print("children of ", doc[i].text)
            for child in doc[i].children:
                # print(child.text, child.i)
                matrix[i][child.i] = self.DEP_MAP[child.dep_]
            # print()

            # in dependency arcs
            matrix[doc[i].head.i][i] = self.DEP_MAP[doc[i].dep_]

        return matrix


if __name__ == "__main__":
    s = "A bilateral retrobulbar neuropathy with an unusual central bitemporal hemianopic scotoma was found"
    builder = AdjMatrixBuilder()
    m = builder(gen_dependency_tree(s))
    print(m)
