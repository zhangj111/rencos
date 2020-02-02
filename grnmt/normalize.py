from simi import simi, read_data, write_data
import sys

def modify_scores(filename, length=10000):

    sources = read_data('%s/test/test.spl.src'%root)
    refs = read_data('%s/test/'%root+filename)
    with open('%s/test/prs.m'%root, 'w') as fw:
        for i, (s, r) in enumerate(zip(sources, refs)):
            print(i)
            s,r = ' '.join(s.split()[:length]),' '.join(r.split()[:length])
            score = simi(r, s, False)
            fw.write('%.2f\n'%score)


if __name__ == '__main__':
    lang = sys.argv[1]
    root = '../samples/%s' % lang
    src_len = 0
    if lang == 'python':
        src_len = 100
    elif lang == 'java':
        src_len = 300
    else:
        print("Wrong argument.")
        exit(-1)
    modify_scores("test.ref.src", length=src_len)
