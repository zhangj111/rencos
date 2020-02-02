import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def nltk_corpus_bleu(hypotheses, references, order=4):
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction
    refs = []
    count = 0
    total_score = 0.0

    cc = SmoothingFunction()

    for hyp, ref in zip(hypotheses, references):
        hyp = hyp.split()
        ref = ref.split()
        refs.append([ref])

        while len(hyp) <= 1:
            hyp += ['.']

        score = nltk.translate.bleu([ref], hyp, smoothing_function=cc.method4)
        total_score += score
        count += 1
    hypotheses = [h.split() if isinstance(h, str) else h for h in hypotheses]
    avg_score = total_score / count
    corpus_bleu = nltk.translate.bleu_score.corpus_bleu(refs, hypotheses)
    # print('corpus_bleu: %.4f avg_score: %.4f' % (corpus_bleu, avg_score))
    return corpus_bleu, avg_score


def main():
    with open('samples/%s/train/train.spl.src'%lang, 'r') as tr:
        corpus = [' '.join(line.split()[:max_len]) for line in tr.readlines()]

    with open('samples/%s/train/train.spl.src'%lang, 'r') as tr:
        sources = tr.readlines()

    def tokenizer(text):
        return text.split()

    vectorizer = CountVectorizer(tokenizer=tokenizer, lowercase=False)

    transformer = vectorizer
    matrix = transformer.fit_transform(corpus)

    with open('samples/%s/test/test.spl.src'%lang, 'r') as te:
        queries = [' '.join(line.split()[:max_len]) for line in te.readlines()]
    with open('samples/%s/train/train.txt.tgt'%lang, 'r') as tr:
        summaries = tr.readlines()

    refs = []
    k = 5
    cosine_similarities_list = cosine_similarity(transformer.transform(queries), matrix)
    for i in range(len(cosine_similarities_list)):
        cosine_similarities = cosine_similarities_list[i]
        pr = 0
        best_bleu = 0
        for j in range(k):
            index = cosine_similarities.argmax()
            src = sources[index].strip()
            qry = queries[i].strip()
            _, bleu = nltk_corpus_bleu([qry], [src])
            if bleu > best_bleu:
                best_bleu = bleu
                pr = index
            cosine_similarities[index] = -float('inf')

        refs.append(summaries[pr].strip() + '\n')
        print(i)

    with open('samples/%s/output/nngen.out'%lang, 'w') as fw:
        fw.writelines(refs)


if __name__ == '__main__':
    lang = sys.argv[1]
    assert lang in ['python', 'java']
    if lang == 'python':
        max_len = 100
    else:
        max_len = 300
    main()
