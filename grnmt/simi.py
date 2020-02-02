import collections
import os
import sys
import math
import numpy as np


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

    # print matrix[n-1][m-1]
    # print matrix
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

#print ("simi score: " + str(simi_score))


def ranker(input_sentence, list_of_sentences, is_list):
    best_score = 0.0
    best_sentence = ""
    for sentence in list_of_sentences:
        simi_score = simi(input_sentence, sentence, is_list)
        if simi_score > best_score:
            best_score = simi_score
            best_sentence = sentence

    return best_score, best_sentence


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

if __name__ == "__main__":

    first = "a x b c e f"
    second = "a b d e v w x y"
    #first = "NAME_BEGIN Assassin 's Blade NAME_END ATK_BEGIN 3 ATK_END DEF_BEGIN -1 DEF_END COST_BEGIN 5 COST_END DUR_BEGIN 4 DUR_END TYPE_BEGIN Weapon TYPE_END PLAYER_CLS_BEGIN Rogue PLAYER_CLS_END RACE_BEGIN NIL RACE_END RARITY_BEGIN Common RARITY_END DESC_BEGIN NIL DESC_END"
    #second = "NAME_BEGIN Mark of the Wild NAME_END ATK_BEGIN -1 ATK_END DEF_BEGIN -1 DEF_END COST_BEGIN 2 COST_END DUR_BEGIN -1 DUR_END TYPE_BEGIN Spell TYPE_END PLAYER_CLS_BEGIN Druid PLAYER_CLS_END RACE_BEGIN NIL RACE_END RARITY_BEGIN Free RARITY_END DESC_BEGIN Give a minion Taunt and +2/+2 . ( +2 Attack/+2 Health ) DESC_END"
    print("here")
    first_l = first.split(" ")
    second_l = second.split(" ")
    matrix, dist = sentence_distance(first_l, second_l, True)
    print("distance: " + str(dist))
    matrix, dist = sentence_distance(first, second, False)
    print("distance: " + str(dist))
    align(first, second, matrix, True, False)

    # sent_list = read_data(
    #     "/Users/shayati/Documents/sem2/NN for NLP/project/hs_dataset_real/hearthstone/train_hs.in")
    # query = "Magma Rager NAME_END 5 ATK_END 1 DEF_END 3 COST_END -1 DUR_END Minion TYPE_END Neutral PLAYER_CLS_END NIL RACE_END Free RARITY_END NIL"
    # score, sentence = ranker(query, sent_list, False)
    # print("ranker")
    # print(score)
    # print(sentence)
    # matrix, dist = sentence_distance(query, sentence, False)
    # #print("distance: " + str(dist))
    # align(query, sentence, matrix, True, False)
    # print(dist)