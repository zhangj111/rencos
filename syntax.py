import sys
import lucene
import re
from java.nio.file import Paths
from java.lang import Integer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.document import Document, Field, StringField, TextField, StoredField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig,IndexOptions,DirectoryReader
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher, BooleanQuery
from org.apache.lucene.queryparser.classic import QueryParser

lucene.initVM()


def build_index(file_dir):
    indexDir = SimpleFSDirectory(Paths.get(file_dir+"/lucene_index/"))
    config = IndexWriterConfig(WhitespaceAnalyzer())
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    writer = IndexWriter(indexDir, config)

    # t1 = FieldType()
    # t1.setStored(True)
    # t1.setTokenized(False)
    # t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS)
    #
    # t2 = FieldType()
    # t2.setStored(True)
    # t2.setTokenized(True)
    # t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

    print("%d docs in index" % writer.numDocs())
    if writer.numDocs():
        print("Index already built.")
        return
    with open(file_dir+"/train/train.ast.src") as fc:

        codes = [re.sub("[\W\s]+|AND|NOT|OR", ' ', line.strip()) for line in fc.readlines()]

    for k, code in enumerate(codes):
        doc = Document()
        doc.add(StoredField("id", str(k)))
        doc.add(TextField("code", code, Field.Store.YES))

        writer.addDocument(doc)

    print("Closing index of %d docs..." % writer.numDocs())
    writer.close()


def retriever(file_dir):
    analyzer = WhitespaceAnalyzer()
    reader = DirectoryReader.open(SimpleFSDirectory(Paths.get(file_dir+"/lucene_index/")))
    searcher = IndexSearcher(reader)
    queryParser = QueryParser("code", analyzer)
    BooleanQuery.setMaxClauseCount(Integer.MAX_VALUE)

    with open(file_dir + "/train/train.spl.src", 'r') as fso,  open(file_dir + "/train/train.txt.tgt", 'r') as fsu:
        sources = [line.strip() for line in fso.readlines()]
        summaries = [line.strip() for line in fsu.readlines()]
    with open(file_dir+"/test/test.ast.src") as ft, open(file_dir+"/test/test.ref.src.0", 'w') as fwo, \
            open(file_dir+"/output/ast.out", 'w') as fws:
        queries = [re.sub("[\W\s]+|AND|NOT|OR", ' ', line.strip()) for line in ft.readlines()]

        for i, line in enumerate(queries):
            print("query %d" % i)
            query = queryParser.parse(QueryParser.escape(line))
            hits = searcher.search(query, 1).scoreDocs
            flag = False

            for hit in hits:
                doc = searcher.doc(hit.doc)
                _id = eval(doc.get("id"))
                flag = True
                fwo.write(sources[_id]+'\n')
                fws.write(summaries[_id] + '\n')
            if not flag:
                print(query)
                print(hits)
                exit(-1)


if __name__ == '__main__':
    root = 'samples/%s'%sys.argv[1]

    build_index(root)
    retriever(root)
