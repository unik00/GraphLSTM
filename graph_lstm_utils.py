import numpy as np

from emb_utils import gen_dependency_tree
import tensorflow as tf


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

    def __call__(self, doc, return_weighted=True):
        """
        Args:
            doc: spacy doc

        Returns:
            2D numpy array denoting adjacency matrix
        """
        matrix = np.zeros(dtype=np.int16, shape=(len(doc), len(doc), 2))

        for i in range(len(doc)):
            # self arc
            if return_weighted:
                matrix[i][i][0] = self.SELF_ARC
            else:
                matrix[i][i][0] = 1
            # next arc
            if i < len(doc) - 1:
                if return_weighted:
                    matrix[i][i + 1][0] = self.NEXT_ARC
                else:
                    matrix[i][i + 1][0] = 1

        for i in range(len(doc)):
            if return_weighted:
                matrix[doc[i].head.i][i][1] = self.DEP_MAP[doc[i].dep_]
            else:
                matrix[doc[i].head.i][i][1] = 1

        return tf.constant(matrix, dtype=tf.float32)


if __name__ == "__main__":
    s = "A bilateral retrobulbar neuropathy with an unusual central bitemporal hemianopic scotoma was found"
    builder = AdjMatrixBuilder()
    m = builder(gen_dependency_tree(s))
    # print(m)
