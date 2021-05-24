import bioc

# TODO: get vocab
if __name__ == "__main__":
    filename = "BioCreative-V-CDR-Corpus/CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.BioC.xml"
    # read from a file
    # read from a ByteIO
    with open(filename, 'rb') as fp:
        reader = bioc.BioCXMLDocumentReader(fp)
        collection_info = reader.get_collection_info()
        for document in reader:
            # process document
            print(document)