import collections
import os
import sys
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing


def sentence_distance(first_sentence, second_sentence, is_list):
    if not is_list:
        first_sentence = first_sentence.split(' ')
        second_sentence = second_sentence.split(' ')
    m = len(first_sentence)+1
    n = len(second_sentence)+1
    matrix = np.zeros((n, m), dtype=int)

    for i in range(n):
        matrix[i][0] = i  # n rows

    for i in range(m):
        matrix[0][i] = i  # m columns
    # print("-----------")
    for i in range(1, n):
        for j in range(1, m):
            if first_sentence[j-1] == second_sentence[i-1]:
                penalty = 0
            else:
                penalty = 1

            # get min
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1] + penalty)

    # print(matrix[n-1][m-1]
    # print(matrix
    return matrix, matrix[n-1][m-1]


def align(first_sentence, second_sentence, matrix, print_flag, is_list):
    if not is_list:
        first_sentence = first_sentence.split(' ')
        second_sentence = second_sentence.split(' ')
    m = len(first_sentence)
    n = len(second_sentence)
    first_index_dict = {}
    second_index_dict = {}
    i = n
    j = m
    reverse1 = []
    reverse2 = []
    #print("first sentence: " + str(first_sentence) + " " + str(len(first_sentence)))
    #print("second sentence: " + str(second_sentence) + " " + str(len(second_sentence)))
    #print("i: " + str(i))
    #print("j: " + str(j))
    unedited_words = {}
    while i != 0 and j != 0:
        first_index = j-1
        second_index = i-1
        #print("first index: " + str(first_index))
        #print("second index: " + str(second_index))
        word1 = first_sentence[first_index]
        word2 = second_sentence[second_index]
        same_words = (first_sentence[first_index] == second_sentence[second_index])
        if same_words:
            unedited_words[word1] = first_index
            temp_value = matrix[i-1][j-1]
        else:
            temp_value = matrix[i-1][j-1] + 1

        if matrix[i][j] == temp_value:
            i = i-1
            j = j-1
            reverse1.append(word1)
            reverse2.append(word2)
            first_index_dict[first_index] = [word1, word2, same_words, second_index]
            second_index_dict[second_index] = [word1, word2, same_words, first_index]
        elif matrix[i][j] == matrix[i-1][j] + 1:
            #print("2nd case -- i: " + str(i) + " j: " + str(j))
            reverse1.append("-")
            reverse2.append(word2)
            i = i-1
            first_index_dict[first_index] = ["NULL", word2, same_words, None]
            second_index_dict[second_index] = ["NULL", word2, same_words, None]
        else:
            #print("3rd case -- i: " + str(i) + " j: " + str(j) + " sec idx: " + str(second_index))
            reverse1.append(word1)
            reverse2.append("-")
            j = j-1
            first_index_dict[first_index] = [word1, "NULL", same_words, None]
            second_index_dict[second_index] = [word1, "NULL", same_words, None]
            # print("--------------------")
    # completing
    for index in range(i):
        second_index_dict[index] = ["NULL", second_sentence[index], False, None]
    # or...
    for index in range(j):
        first_index_dict[index] = [first_sentence[index], "NULL", False, None]

    if print_flag:

        print("----- ALIGNMENT ---- ")
        print(first_sentence)
        print(second_sentence)
        print(reverse1[::-1])
        print(reverse2[::-1])
        print("----- END OF ALIGNMENT ---- ")
        print(unedited_words)
        print(first_index_dict)
        print(second_index_dict)
    return unedited_words, first_index_dict, second_index_dict

#	x = first_sentence
#	x_m = second_sentence


def simi(first_sentence, second_sentence, is_list):
    if not is_list:
        max_score = max(len(first_sentence.split()), len(second_sentence.split()))
        simi = 1.0 - (float(sentence_distance(first_sentence,
                                              second_sentence, False)[1])/float(max_score))
    else:
        max_score = max(len(first_sentence), len(second_sentence))
        simi = 1.0 - (float(sentence_distance(first_sentence,
                                              second_sentence, True)[1])/float(max_score))
    return simi


def score(pair):
    input_sentence, sentence, index = pair
    simi_score = simi(input_sentence, sentence, False)
    return index, simi_score


def ranker(input_sentence, list_of_sentences, is_list):
    best_score = 0.0
    best_sentence = ""
    best_index = 0

    input_sentences = [input_sentence]*len(list_of_sentences)
    pairs = zip(input_sentences, list_of_sentences, list(range(len(list_of_sentences))))

    results = pool.map(score, pairs)

    for p in results:

        simi_score = p[1]
        if simi_score > best_score:
            best_score = simi_score
            best_index = p[0]
    best_sentence = list_of_sentences[best_index]
    # for i, sentence in enumerate(list_of_sentences):
    #     # if abs(len(sentence.split()) - input_len) > 10:
    #     #     continue
    #     simi_score = simi(input_sentence, sentence, is_list)
    #     if simi_score > best_score:
    #         best_score = simi_score
    #         best_sentence = sentence
    #         best_index = i

    return best_score, best_sentence, best_index


def read_data(filename):
    sentence_list = []
    with open(filename, "r") as f:
        for line in f:
            sentence_list.append(line.strip())
    return sentence_list


def write_data(data, filename):
    with open(filename, "w") as f:
        for item in data:
            f.write(item+'\n')

def load_documents():

    corpus = read_data('samples/python/train/train.ast.src')

    sources = read_data('samples/python/train/train.spl.src')

    # def tokenizer(text):
    #     return re.split(r'[^A-Za-z_]+', text)

    vectorizer = TfidfVectorizer(tokenizer=None)
    tfidf = vectorizer.fit_transform(corpus)
    # with open('samples/python/test/test.ast.src', 'r') as te:
    #     queries = te.readlines()
    with open('samples/python/train/train.txt.tgt', 'r') as tr:
        summaries = tr.readlines()

    return (tfidf, vectorizer), (corpus, sources, summaries)


def re_ranker(queries):
    import time
    hypothesis = []
    prs = []
    refs = []
    k = 10
    cosine_similarities_list = cosine_similarity(vectorizer.transform(queries), tfidf)
    for i in range(len(cosine_similarities_list)):
        cosine_similarities = cosine_similarities_list[i]

        indexes = np.argpartition(cosine_similarities, -k)[-k:]

        documents = [corpus[index].strip() for index in indexes]

        score, sentence, position = ranker(queries[i], documents, False)

        index = indexes[position]
        prs.append('%.2f\n' % score)
        hypothesis.append(sources[index].strip() + '\n')
        refs.append(summaries[index].strip() + '\n')
        print(i)

    with open('samples/python/output/test.out', 'w') as fw:
        fw.writelines(refs)
    #
    # with open('samples/python/test/test.ref.src.%d' % (k - 1), 'w') as fw:
    #     fw.writelines(hypothesis)
    #
    # with open('samples/python/test/prs.%d' % (k - 1), 'w') as fw:
    #     fw.writelines(prs)

if __name__ == "__main__":

    queries = read_data('samples/python/test/test.ast.src')
    raws = read_data('samples/python/test/test.spl.src')
    (tfidf, vectorizer), (corpus, sources, summaries) = load_documents()
    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        re_ranker(queries)
