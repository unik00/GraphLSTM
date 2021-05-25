import bioc
from os import path

from emb_utils import gen_dependency_tree


def normalize(s: str):
    return s.lower()


def load_built_data(filepath):
    ret = dict()
    with open(filepath, "r") as f:
        for i, val in enumerate(f.readlines()):
            ret[val.strip("\n")] = i + 1
    return ret


def write_dict_down(d, filepath):
    with open(filepath, "w") as f:
        for key in d.keys():
            f.write(key + "\n")


class CDRData:
    DEFAULT_VOCAB_PATH = "saved_datas/train_vocab.txt"
    DEFAULT_CHAR_DICT_PATH = "saved_datas/char_dict.txt"

    def build_vocab_from_raw_data(self, filename, write_down=True):
        ret = dict()

        with open(filename, 'rb') as fp:
            reader = bioc.BioCXMLDocumentReader(fp)
            collection_info = reader.get_collection_info()
            for document in reader:
                # process document
                for passage in document.passages:
                    print(passage.text)
                    dependency = gen_dependency_tree(passage.text)
                    for token in dependency:
                        ret[normalize(token.text)] = 1

        cnt = 0
        for key in ret:
            cnt += 1
            ret[key] = cnt

        if write_down:
            write_dict_down(ret, self.DEFAULT_VOCAB_PATH)

        return ret

    def char_dict_from_vocab(self, vocab_dict, write_down=True):
        ret = dict()
        for key in vocab_dict:
            for c in key:
                ret[c] = 1

        cnt = 0
        for key in ret:
            cnt += 1
            ret[key] = cnt

        if write_down:
            write_dict_down(ret, self.DEFAULT_CHAR_DICT_PATH)

        return ret

    def build_train_vocab(self):
        m1 = self.build_vocab_from_raw_data(
            "BioCreative-V-CDR-Corpus/CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.BioC.xml")
        m2 = self.build_vocab_from_raw_data(
            "BioCreative-V-CDR-Corpus/CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.BioC.xml")
        m = {**m1, **m2}
        return m

    def __init__(self, build_vocab=False):
        if not path.exists(self.DEFAULT_VOCAB_PATH) \
                or not path.exists(self.DEFAULT_CHAR_DICT_PATH) \
                or build_vocab:
            self.vocab_dict = self.build_train_vocab()
            self.char_dict = self.char_dict_from_vocab(self.vocab_dict)
        else:
            self.vocab_dict = load_built_data(self.DEFAULT_VOCAB_PATH)
            self.char_dict = load_built_data(self.DEFAULT_CHAR_DICT_PATH)


if __name__ == "__main__":
    cdr_data = CDRData()
    print(cdr_data.vocab_dict)
    print(cdr_data.char_dict)