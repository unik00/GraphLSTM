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


def build_inter_sentence_docs_from_file(path, limit=None):
    """
    Args:
        path: a string denoting a path

    Returns:
        A list of dicts. One dict should contain the following keys:
            - "doc": spacy docs
            - "Disease": a dict containing diseases as keys. The value of each key is
                a list containing one pair of starting end ending token index
            - "Chemical": the same as "disease"
            - "relation": a set of pairs (chemical, disease) denoting one CID relation
    """
    ret = list()

    with open(path, 'rb') as fp:
        reader = bioc.BioCXMLDocumentReader(fp)

        for document in reader:
            current_dict = dict()
            current_dict['Disease'] = dict()
            current_dict['Chemical'] = dict()

            for passage in document.passages:
                assert len(passage.relations) == 0
                doc = gen_dependency_tree(passage.text)
                current_dict['doc'] = doc

                for annotation in passage.annotations:
                    MESH = annotation.infons.get("MESH")
                    typ = annotation.infons.get("type")
                    if MESH not in current_dict[typ]:
                        current_dict[typ][MESH] = list()

                    # convert character location to token location
                    converted_loc = list()
                    for location in annotation.locations:
                        left = -1
                        right = -1

                        for i, token in enumerate(doc):
                            if location.offset - passage.offset >= token.idx and \
                                    location.offset + location.length - passage.offset <= token.idx + len(token):
                                # In case entity is within token. e.g. "carcinogenic" in "Co-carcinogenic"
                                left = i
                                right = i
                                break

                            if token.idx >= location.offset - passage.offset and left == -1:
                                left = i
                            if token.idx < location.offset + location.length - passage.offset:
                                right = i
                        converted_loc.append((left, right))
                    current_dict[typ][MESH] += converted_loc

            current_dict["relation"] = set()
            for relation in document.relations:
                assert len(relation.nodes) == 0
                assert relation.infons.get("relation") == "CID"
                current_dict["relation"].add((relation.infons.get("Chemical"),
                                              relation.infons.get("Disease")))

            ret.append(current_dict)
            if limit is not None and len(ret) == limit:
                return ret

    return ret


def build_intra_sentence_docs_from_file(path, limit=None):
    """ Refer to build_intra_sentence_docs_from_file() docstring for arguments explanation """
    # build inter sentence level first
    inter_list = build_inter_sentence_docs_from_file(path)

    # convert inter-sentence to intra sentence
    new_list = list()
    for old_dict in inter_list:
        old_doc = old_dict["doc"]
        for sent in old_doc.sents:
            new_dict = dict()
            new_dict["doc"] = sent
            new_dict["Chemical"] = dict()
            new_dict["Disease"] = dict()
            new_dict["relation"] = set()

            for typ in ["Chemical", "Disease"]:
                for MESH in old_dict[typ]:
                    for pos in old_dict[typ][MESH]:
                        if pos[0] >= sent.start and pos[1] < sent.end:
                            if MESH not in new_dict[typ]:
                                new_dict[typ][MESH] = list()
                            new_dict[typ][MESH].append(pos)

            for relation in old_dict["relation"]:
                if relation[0] in new_dict["Chemical"] and relation[1] in new_dict["Disease"]:
                    new_dict["relation"].add(relation)

            new_list.append(new_dict)
            if limit is not None and len(new_list) == limit:
                return new_list
    return new_list


class CDRData:
    DEFAULT_VOCAB_PATH = "saved_datas/train_vocab.txt"
    DEFAULT_CHAR_DICT_PATH = "saved_datas/char_dict.txt"
    DEV_DATA_PATH = "BioCreative-V-CDR-Corpus/CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.BioC.xml"
    TRAIN_DATA_PATH = "BioCreative-V-CDR-Corpus/CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.BioC.xml"

    def build_vocab_from_raw_data(self, filename, write_down=True):
        ret = dict()

        with open(filename, 'rb') as fp:
            reader = bioc.BioCXMLDocumentReader(fp)
            collection_info = reader.get_collection_info()
            for document in reader:
                print("document: ", document)
                for passage in document.passages:
                    assert (len(passage.relations) == 0)
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
            self.DEV_DATA_PATH)
        m2 = self.build_vocab_from_raw_data(
            self.TRAIN_DATA_PATH)
        m = {**m1, **m2}
        return m

    def build_data_from_file(self, path, mode, limit=None):
        """
        Args:
            path: a string
            mode: either "inter" or "intra"
            limit: a integer, limit of data length (mainly for debug)
        """
        if mode == "inter":
            return build_inter_sentence_docs_from_file(path, limit)
        elif mode == "intra":
            return build_intra_sentence_docs_from_file(path, limit)

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
    cdr_data = CDRData(build_vocab=False)
    # print(cdr_data.vocab_dict)
    # print(cdr_data.char_dict)
    cdr_data.build_data_from_file(cdr_data.DEV_DATA_PATH, mode="intra")
