# -*- coding: UTF-8 -*-
from __future__ import unicode_literals

import copy
import json
import my_linked_list as mll
import heapq
from hazm import *
from math import log, pow
from styles_utility import Styles

with open("IR_data_news_12k.json", "r") as read_file:
    data = json.loads(read_file.read())

normalizer = Normalizer()
stemmer = Stemmer()
tokenizer = WordTokenizer()
lemmatizer = Lemmatizer()
doc_count = 0
dictionary: {str, mll.LinkedList} = dict()
champ_dict: {str, mll.LinkedList} = dict()
docs_tokens: list[list[int]]
docs_tokens_sets = [set[str]]
normalization_factors: [float]
HIGH_IDF_TERMS_COUNT = 5
K_TOP_DOCS: int = 5
TOP_IDF_QUERY_TERMS = 4
CHAMP_LIST_SIZE: int = 75


def check_system(collection) -> None:
    """check to see if the systems work"""
    for i in range(50):
        print(f'URL: {collection[str(i)]["url"]}\ntitle: {collection[str(i)]["title"]}')


def build_tokens_lists(collection):
    """build tokens lists"""
    global docs_tokens, doc_count, normalization_factors
    docs_tokens = list()
    print(f"number of documents: {len(data.keys())}")
    print("enter the number of documents to be processed.")
    num = int(input())
    for i in range(num):
        single_news = str((collection[str(i)]['content']))
        normalized = normalizer.normalize(single_news)
        tokenized = word_tokenize(normalized)
        remove_stopwords_and_punctuations(tokenized)
        stemmed = [lemmatizer.lemmatize(stemmer.stem(x)) for x in tokenized]
        docs_tokens.append(stemmed)
    # remove_stopwords_and_punctuations()
    doc_count = len(docs_tokens)
    normalization_factors = [0.0] * len(docs_tokens)


def remove_stopwords_and_punctuations(li: [str]) -> None:
    """remove stopwords and punctuations"""
    stop_words = set(stopwords_list())
    punctuations = [',', ';', '،', '.', '[', ']', '؛', '(', ')', '{', '}', '?', ':']
    # for li in docs_tokens:
    for i in range(0, len(li)):
        if li[i] in stop_words:
            li[i] = ''
        elif li[i] in punctuations:
            li[i] = '\0'


def build_dict() -> None:
    """build an unordered dictionary based on hash"""
    global dictionary
    temp_dict = dict()
    i = 0
    for doc_tokens in docs_tokens:
        # `j` is the current index of the word in a document
        j = -1
        for token in doc_tokens:
            j += 1
            if token == '':
                continue
            elif token == '\0':
                # j -= 1
                continue
            # insert into temp_dict if not available to later be inserted into the final dictionary
            if temp_dict.get(token) is None:
                temp_dict[token] = mll.Node(i)
            # insert index into doc's node
            temp_dict[token].indexes.append(j)
        for word in temp_dict:
            # merging the temporary dictionary into the main dictionary
            if dictionary.get(word) is None:
                dictionary[word] = mll.LinkedList()
            dictionary[word].insert(temp_dict[word])
        temp_dict.clear()
        i += 1


def retrieve_docs(word: str) -> mll.LinkedList:
    """return docs in which the word is present"""
    if dictionary.get(word):
        return dictionary[word]
    return mll.LinkedList()


def is_subarray(arr1, arr2) -> bool:
    """checks if arr2 is subarray of arr1"""
    i, j = 0, 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] == arr2[j]:
            j += 1
            if j == len(arr2):
                return True
        else:
            j = 0
        i += 1
    return False


def ll_to_arr(ll: mll.LinkedList) -> list[list[int]]:
    """get a linked list and returns back an array"""
    arr = [[], []]
    curr = ll.head
    while curr:
        arr[0].append(curr.doc_id)
        arr[1].append(curr.count)
        curr = curr.next
    return arr


def arr_to_ll(array: list[list[int]]) -> mll.LinkedList:
    """convert a 2d array into a linked list array[0] is filled
    with doc_id and array[1] is filled with number of repetitions"""
    newList = mll.LinkedList()
    for i in range(len(array[0])):
        newNode = mll.Node(array[0][i])
        newNode.indexes = array[1][i] * [0]  # list is filled with 0 (garbage), this for the sake of correct size
        newList.insert(newNode)
    return newList


def or_docs(list1: mll.LinkedList, list2: mll.LinkedList) -> mll.LinkedList:
    """process `AND`"""
    node1 = list1.head
    node2 = list2.head
    final_list = [[], []]
    while node1 and node2:
        if node1.doc_id == node2.doc_id:
            final_list[0].append(node1.doc_id)
            final_list[1].append(node1.count + node2.count)
            node1 = node1.next
            node2 = node2.next
        elif node1.doc_id > node2.doc_id:
            final_list[0].append(node2.doc_id)
            final_list[1].append(node2.count)
            node2 = node2.next
        else:
            final_list[0].append(node1.doc_id)
            final_list[1].append(node1.count)
            node1 = node1.next

    if not node1:
        while node2:
            final_list[0].append(node2.doc_id)
            final_list[1].append(node2.count)
            node2 = node2.next
    else:
        while node1:
            final_list[0].append(node1.doc_id)
            final_list[1].append(node1.count)
            node1 = node1.next

    return arr_to_ll(final_list)


def and_docs(list1: mll.LinkedList, list2: mll.LinkedList) -> mll.LinkedList:
    """process `AND`"""
    node1 = list1.head
    node2 = list2.head
    final_list = [[], []]
    while node1 and node2:
        if node1.doc_id == node2.doc_id:
            # this list must be sorted
            final_list[0].append(node1.doc_id)
            final_list[1].append(node1.count + node2.count)
            node1 = node1.next
            node2 = node2.next
        elif node1.doc_id > node2.doc_id:
            node2 = node2.next
        else:
            node1 = node1.next
    return arr_to_ll(final_list)


def not_and_docs(list1: mll.LinkedList, list2: mll.LinkedList) -> mll.LinkedList:
    """process `! AND`"""
    node1 = list1.head
    node2 = list2.head
    final_list = [[], []]
    while node1:
        if node1 and not node2 or node1.doc_id < node2.doc_id:
            final_list[0].append(node1.doc_id)
            final_list[1].append(node1.count)
            node1 = node1.next
        elif node1.doc_id == node2.doc_id:
            node1 = node1.next
            node2 = node2.next
        else:
            node2 = node2.next
    return arr_to_ll(final_list)


def get_expression_docs(words: list[str]):
    """return the documents which consist the expression"""
    ll_docs: mll.LinkedList = dictionary[words[0]]
    curr = ll_docs.head
    docs_counts, count, j = [[], []], 0, 0
    while curr:
        for index in curr.indexes:
            if docs_tokens[curr.doc_id][index: index + len(words)] == words:
                count += 1
        if count != 0:
            docs_counts[0].append(curr.doc_id)
            docs_counts[1].append(count)
        curr = curr.next
        count = 0
    return arr_to_ll(docs_counts)


def process_query(query: str) -> tuple[set[str], list[int], set[str]]:
    """process a query and return the result of the query as a list of document ids"""
    parsed_query = parse_query(query)
    print(f"parsed query: {parsed_query}")
    count = len(docs_tokens)
    curr_res = [range(0, count), count * [0]]
    curr_res = arr_to_ll(curr_res)
    curr_res_backup = arr_to_ll([[], []])
    tokens = set()
    i = 0
    while i < len(parsed_query):
        if parsed_query[i] == "!":  # processing NOT operator
            i += 1
            if parsed_query[i] == "$":
                i += 1
                words = []  # words in the expression
                while parsed_query[i] != "$":
                    words.append(parsed_query[i])
                    parsed_query[i] = "none"  # i did this so in the future i dont give score to these terms
                    i += 1
                ll = get_expression_docs(words)
            else:
                if not dictionary[parsed_query[i]]:
                    continue
                ll = dictionary[parsed_query[i]]
                parsed_query[i] = "none"  # i did this so in the future i dont give score to these terms
            curr_res = not_and_docs(curr_res, ll)
            curr_res_backup = not_and_docs(curr_res_backup, ll)
            i += 1
        elif parsed_query[i] == "$":  # processing expression
            i += 1
            words = []  # words in the expression
            while parsed_query[i] != "$":
                words.append(parsed_query[i])
                i += 1
            i += 1
            ll = get_expression_docs(words)
            curr_res = and_docs(curr_res, ll)
            curr_res_backup = or_docs(curr_res_backup, ll)
            expression = "$"
            for word in words:
                expression += word + " "
            tokens.add(expression)
        elif not dictionary.get(parsed_query[i]):  # term is not in the dictionary
            i += 1
            continue
        else:  # processing AND operator
            tokens.add(parsed_query[i])
            ll = dictionary[parsed_query[i]]
            curr_res = and_docs(curr_res, ll)
            curr_res_backup = or_docs(curr_res_backup, ll)
            i += 1
    final_list = ll_to_arr(curr_res)
    final_sorted = [y for _, y in sorted(zip(final_list[1], final_list[0]), reverse=True)]
    print(f"normal: {final_sorted}")
    if curr_res.size < 5:  # in case we have less than 5 documents
        remove_repetitives(curr_res, curr_res_backup)
        final_list_backup = ll_to_arr(curr_res_backup)
        final_backup_sorted = [y for _, y in sorted(zip(final_list_backup[1], final_list_backup[0]),
                                                    reverse=True)]
        final_sorted.extend(final_backup_sorted)
        print(f"backup: {final_backup_sorted}")
    if not len(tokens):  # if no term was in the dictionary return an empty list of doc ids
        final_sorted = []
    parsed_query = set(parsed_query)
    if "!" in parsed_query:
        parsed_query.remove("!")
    if "$" in parsed_query:
        parsed_query.remove("$")
    if "none" in parsed_query:
        parsed_query.remove("none")
    return tokens, final_sorted, parsed_query


def remove_repetitives(res: mll.LinkedList, backup: mll.LinkedList):
    """merge the result of OR and AND"""
    res_curr = res.head
    backup_curr = backup.head
    while backup_curr and res_curr:
        if backup_curr.doc_id == res_curr.doc_id:
            backup.remove(backup_curr)
            backup_curr = backup_curr.next
            res_curr = res_curr.next
        elif backup_curr.doc_id < res_curr.doc_id:
            backup_curr = backup_curr.next
        else:
            res_curr = res_curr.next


def parse_query(query: str) -> list[str]:
    """break the query into smaller queries"""
    x = query.strip()
    y = x.split(" ")
    i, word, z = 0, "", []
    while i in range(len(y)):
        if y[i][0] == "\"":
            z.append("$")  # '$' indicates start and end of the expression
            if y[i][-1] != "\"":
                z.append(y[i][1:])
                i += 1
                while y[i][-1] != "\"":
                    z.append(y[i])
                    i += 1
                z.append(y[i][:-1])
            else:
                z.append(y[i][1:-1])
            z.append("$")
        else:
            z.append(y[i])
        i += 1
    normalized = [normalizer.normalize(x) for x in z]
    stemmed = [stemmer.stem(x) for x in normalized]
    lemmatized = [lemmatizer.lemmatize(x) for x in stemmed]
    final = [x for x in lemmatized if x != '']
    return final


def retrieve_sentences(docs: list[int], tokens: set[str]) -> None:
    """print the sentences in which required words exist"""
    if len(docs) > 5:  # if there are many documents retrieve only first 5
        docs = docs[0:5]
    for doc_id in docs:
        title = data[str(doc_id)]['title']
        print(f'{Styles.OKBLUE}document id: {doc_id}\ntitle: {title}\nsentences: ')
        sentences = sent_tokenize(data[str(doc_id)]['content'])
        for sent in sentences:
            normalized = normalizer.normalize(sent)
            tokenized = word_tokenize(normalized)
            stemmed = [stemmer.stem(x) for x in tokenized]
            lemmatized = [lemmatizer.lemmatize(x) for x in stemmed]
            if doc_id == 160:
                print(sent)
            if any(x in lemmatized for x in tokens):  # normal words expressions
                print(f"{Styles.OKGREEN}{sent}")
            for token in tokens:  # look for expressions
                if token[0] == "$":
                    words = token[1:].strip().split(" ")
                    # print(f"words: {words}'")
                    if is_subarray(lemmatized, words):
                        print(sent)
    print(f'{Styles.ENDC}')


def get_weight(hit: mll.LinkedList, doc_id: int) -> float:
    """
    Args:
        doc_id: weight of the term is calculated in this document
        hit: is the result of searching the dictionary for a term

    Additional:
        tf: term-frequency
        idf: inverse document frequency

    Returns:
        weight of the term in the document
    """
    global docs_tokens
    node = hit.get(doc_id)
    tf = 0 if not node else node.count
    df = hit.size
    N = len(docs_tokens)
    return ((1 + log(tf, 10)) if tf != 0 else 0) * (log(N / df, 10))


def get_normal_factors(doc_id: int) -> [float]:
    """
    Args:
        doc_id: id of the document whose normalization factor is required
    Returns:
        a list containing vector size of each document for later use in scoring
    """
    global docs_tokens
    if normalization_factors[doc_id] != 0.0:
        return normalization_factors[doc_id]
    total_weight = 0
    for token in docs_tokens[doc_id]:
        if dictionary.get(token):
            hit = dictionary[token]
            total_weight += pow(get_weight(hit, doc_id), 2)
    normalization_factors[doc_id] = pow(total_weight, 0.5)
    return normalization_factors[doc_id]


def build_champion_list(base_dict: dict):
    """
    Args:
        base_dict: normal dictionary
    Returns:
        a dictionary but filled with champion postings list
    """
    global champ_dict, CHAMP_LIST_SIZE
    for term in base_dict:
        max_heap = []
        ll = base_dict[term]
        curr = ll.head
        while curr:
            heapq.heappush(max_heap, curr)
            curr = curr.next
        r = min(CHAMP_LIST_SIZE, ll.size)
        champ_list = heapq.nlargest(r, max_heap,
                                    lambda node: node.count)  # get the r most relevant documents (based of tf)
        champ_list = sorted(champ_list, key=lambda node: node.doc_id)  # sort them based on document id
        champ_ll = mll.LinkedList()
        for i in range(len(champ_list)):
            champ_ll.insert(copy.copy(champ_list[i]))  # a shallow copy from objects
        champ_dict[term] = champ_ll


def jaccard_scores(query_terms: set[str], docs: [int]) -> [int]:
    """
    Args:
        docs: id of the documents retrieved from champion lists
        query_terms: terms in the query
    Returns:
        calculates the jaccard coefficient
    """
    scores = list()
    for doc_id in docs:
        intersection = docs_tokens_sets[doc_id].intersection(query_terms)
        union = docs_tokens_sets[doc_id].union(query_terms)
        scores.append(float(len(intersection)) / len(union))
    return scores


def cosine_score(query: set[str], docs: [int]):
    """
    Args:
        query: tokens found in the query (after being processed in linguistic module)
        docs: documents that match the query after index elimination
    Returns:
        a list of scores for each of the documents in `docs`
    """
    global HIGH_IDF_TERMS_COUNT
    query_dict = get_top_r_idf(query, HIGH_IDF_TERMS_COUNT)
    # N = len(docs_tokens)
    scores = [0] * len(docs)
    for term in query_dict:
        ll = dictionary[term]
        wtq = 1  # (1 + log(query_dict[term])) * (log(N / ll.size, 10))
        for i in range(len(docs)):
            scores[i] += wtq * get_weight(ll, docs[i])

    for i in range(len(docs)):
        scores[i] /= get_normal_factors(docs[i])

    return scores


def get_top_k_docs(docs: [int], scores: [float], k: int) -> [int]:
    """
    Args:
        scores: score list of documents
        docs: documents that match the query after index elimination
        k: number of documents to be retrieved
    Returns:
        list of top k documents
    """
    max_heap = []
    for i in range(len(docs)):
        heapq.heappush(max_heap, (docs[i], scores[i]))
    return ([t[0] for t in heapq.nlargest(k, max_heap, key=lambda tup: tup[1])],
            [t[1] for t in heapq.nlargest(k, max_heap, key=lambda tup: tup[1])])  # remove this


def get_top_r_idf(my_dict: set[str], r: int) -> [str]:
    """
    Args:
        my_dict: initial dictionary built for inverted index
        r: number of highest idf
    Returns:
        a list composed of top r terms with highest idf
    """
    global dictionary
    max_heap = []
    for term in my_dict:
        if term in dictionary:
            heapq.heappush(max_heap, (term, dictionary[term].size))
    return [t[0] for t in heapq.nlargest(r, max_heap, key=lambda tup: tup[1])]


def main():
    global dictionary, docs_tokens, docs_tokens_sets, champ_dict, K_TOP_DOCS
    build_tokens_lists(data)
    for li in docs_tokens:
        docs_tokens_sets.append(set(li))
    build_dict()
    # build_champion_list(dictionary)
    # dictionary = champ_dict
    while True:
        query = input("پرسمان را وارد کنید:")
        tokens_and_docs = process_query(query)

        # print("=================================================")
        # print(f"{Styles.HEADER}COSINE SCORING{Styles.FONT_SIZE_NORMAL}")
        # print(f"terms: {tokens_and_docs[0]}")
        # scores = cosine_score(tokens_and_docs[2], tokens_and_docs[1])
        # print(f'scores:\n{scores}')
        # print(f"before choosing top-k docs:\n{tokens_and_docs[1]}")
        # candidate_docs, candidate_scores = get_top_k_docs(tokens_and_docs[1], scores,
        #                                                   min(K_TOP_DOCS, len(tokens_and_docs[1])))
        # print(f"after choosing top-k docs:\n{candidate_docs}")
        # print(f'scores:\n{candidate_scores}')
        # print([len(docs_tokens[doc]) for doc in candidate_docs])
        # retrieve_sentences(candidate_docs, tokens_and_docs[0])
        retrieve_sentences(tokens_and_docs[1], tokens_and_docs[0])

        # print("=================================================")
        # print(f"{Styles.HEADER}JACCARD SCORING{Styles.FONT_SIZE_NORMAL}")
        # print(f"terms: {tokens_and_docs[0]}")
        # scores = jaccard_scores(tokens_and_docs[2], tokens_and_docs[1])
        # print(f'scores:\n{scores}')
        # print(f"before choosing top-k docs:\n{tokens_and_docs[1]}")
        # candidate_docs, candidate_scores = get_top_k_docs(tokens_and_docs[1], scores,
        #                                                   min(K_TOP_DOCS, len(tokens_and_docs[1])))
        # print(f"after choosing top-k docs:\n{candidate_docs}")
        # print(f'scores:\n{candidate_scores}')
        # print(f'size of docs:\n{[len(docs_tokens[doc]) for doc in candidate_docs]}')
        # retrieve_sentences(candidate_docs, tokens_and_docs[0])


if __name__ == "__main__":
    main()


# def get_top_idf_query_terms(terms: [str]) -> [str]:
#     """
#     Args:
#         terms: terms of the query
#     Returns: top idf terms (number of terms is TOP_IDF_QUERY_TERMS)
#     """
#     idf_scores = []
#     for term in terms:
#         if dictionary.get(term):
#             idf_scores.append(dictionary[term].size)
#     terms = [t for _, t in sorted(zip(idf_scores, terms), key=lambda tup: tup[0])]
#     return terms[0:TOP_IDF_QUERY_TERMS]

# def get_tok_count_dict(query: [str]):
#    """
#    Args:
#        query: query to be indexed
#
#    Returns:
#        returns a dictionary that maps a term in the query to the number of its occurrences
#    """
#    tok_count_dict = {}
#    for token in query:
#        if token in tok_count_dict:
#            tok_count_dict[token] += 1
#        else:
#            tok_count_dict[token] = 1
#    return tok_count_dict


