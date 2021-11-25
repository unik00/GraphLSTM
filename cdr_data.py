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

    DEV_DATA_PATH = "BioCreative-V-CDR-Corpus/CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.BioC.xml"
    TRAIN_DATA_PATH = "BioCreative-V-CDR-Corpus/CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.BioC.xml"
    TEST_DATA_PATH = "BioCreative-V-CDR-Corpus/CDR_Data/CDR.Corpus.v010516/CDR_TestSet.BioC.xml"

    NER_LABEL_TO_ID = {
        'O': 0,
        'BDisease': 1,
        'IDisease': 2,
        'BChemical': 3,
        'IChemical': 4
    }

    def build_inter_sentence_docs_from_file(self, path, limit=None):
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

                    # if passage.infons.get("type") == "title": TODO: ask for elaboration
                    #     continue

                    assert len(passage.relations) == 0
                    doc = gen_dependency_tree(passage.text)
                    current_dict['doc'] = doc
                    current_dict['ner_label'] = [self.NER_LABEL_TO_ID['O'] for _ in range(len(doc))]
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
                        for l, r in converted_loc:
                            current_dict['ner_label'][l] = self.NER_LABEL_TO_ID[f"B{typ}"]
                            for i in range(l + 1, r + 1):
                                current_dict['ner_label'][i] = self.NER_LABEL_TO_ID[f"I{typ}"]

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

    def build_data_from_file(self, path, limit=None):
        """
        Args:
            path: a string
            limit: a integer, limit of data length (mainly for debug)
        """
        return self.build_inter_sentence_docs_from_file(path, limit)

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
    print(len(cdr_data.build_data_from_file(cdr_data.DEV_DATA_PATH)))
