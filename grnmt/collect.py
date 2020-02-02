
from simi import sentence_distance, align
import pickle
import sys


def store_unedited_words(query_file, reference_file):

    with open('./%s/unedited_words.txt' % lang, 'w') as fw, open(query_file, 'r') as fq, open(reference_file, 'r') as fr:
        queries = fq.readlines()
        references = fr.readlines()
        for q, r in zip(queries, references):
            matrix, dist = sentence_distance(q, r, False)
            unedited_words, _, _ = align(q, r, matrix, False, False)
            fw.write(str(unedited_words)+"\n")


def read_word_alignments(align_file, ids_file):
    alignments = []
    with open(align_file, 'r') as fa, open(ids_file) as fi:
        all_aligns = fa.readlines()
        for idx in fi.readlines():
            idx = int(idx)
            alignments.append(all_aligns[idx])
    return alignments


def collect_translation_pieces(source_file, target_file, align_file, ids_file, unedited_file):
    trans_pieces = []
    align_dict = read_word_alignments(align_file, ids_file)
    with open(source_file) as fs, open(target_file) as ft, open(unedited_file) as fu:
        sources = fs.readlines()
        targets = ft.readlines()
        unedited_lines = fu.readlines()

    for s, t, u, a in zip(sources, targets, unedited_lines, align_dict):
        trans_piece = []
        source_words = s.split()
        target_words = t.split()
        unedited_words = eval(u.strip())
        target_length = len(target_words)
        for i in range(target_length):
            for j in range(i, target_length):
                if j-i == 4:
                    break
                flag = False
                for item in a.split():
                    item = item.split('-')
                    k, v = int(item[0]), int(item[1])
                    if v == j and source_words[k] not in unedited_words:
                        flag = True
                if flag:
                    break
                if i != j:
                    trans_piece.append(target_words[i:j])
        trans_pieces.append(trans_piece)
    with open('./%s/translation_pieces.pkl' % lang, 'wb') as fw:
        pickle.dump(trans_pieces, fw)


if __name__ == '__main__':
    lang = sys.argv[1]
    assert lang in ['python', 'java']
    root = '../samples/%s/test/' % lang
    store_unedited_words('%stest.spl.src'%root, '%stest.ref.src'%root)
    collect_translation_pieces(root+'test.ref.src', root+'test.ref.tgt',
                               './%s/train/model/aligned.grow-diag-final-and' % lang,
                               root+'ids.0', './%s/unedited_words.txt' % lang)
