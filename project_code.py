# -*- coding: UTF-8 -*- 
from __future__ import unicode_literals

import sys

from hazm import *
import json
import my_linked_list as mll
import pickle
from dataclasses import dataclass


with open("IR_data_news_12k.json", "r") as read_file:
    data = json.loads(read_file.read())

normalizer = Normalizer()
stemmer = Stemmer()
tokenizer = WordTokenizer()
lemmatizer = Lemmatizer()


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


@dataclass
class Snapshot:
    _dictionary: dict
    _docs_tokens: list[list[int]]

    @property
    def dictionary(self):
        return self._dictionary

    @property
    def docs_tokens(self):
        return self._docs_tokens


def check_system(collection) -> None:
    """check to see if the systems work"""
    for i in range(50):
        print(f'URL: {collection[str(i)]["url"]}\ntitle: {collection[str(i)]["title"]}')


def build_tokens_lists(collection) -> list[list[int]]:
    """build tokens lists"""
    docs_tokens = list()
    print(len(data.keys()))
    for i in range(12202):  # len(data.keys())):  # data.keys():
        single_news = str((collection[str(i)]['content']))  # .encode("utf-8"))
        normalized = normalizer.normalize(single_news)
        tokenized = word_tokenize(normalized)
        stemmed = [lemmatizer.lemmatize(stemmer.stem(x)) for x in tokenized]
        docs_tokens.append(stemmed)
    remove_stopwords_and_punctuations(docs_tokens)
    return docs_tokens


def remove_stopwords_and_punctuations(docs_tokens: list[list[int]]) -> None:
    """remove stopwords and punctuations"""
    stop_words = set(stopwords_list())
    punctuations = [',', ';', '،', '.', '[', ']', '؛', '(', ')', '{', '}', '?', ':']
    for li in docs_tokens:
        for i in range(0, len(li)):
            if li[i] in stop_words:
                li[i] = ''
            elif li[i] in punctuations:
                li[i] = '\0'


def build_dict():
    """build an unordered dictionary based on hash"""
    temp_dict = dict()
    dictionary = dict()
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
    return dictionary


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
        arr[1].append(curr.count())
        curr = curr.next
    return arr


def arr_to_ll(array: list[list[int]]) -> mll.LinkedList:
    """convert a 2d array into a linked list
        array[0] is filled with doc_id and array[1]
        is filled with number of repetitions"""
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
            final_list[1].append(node1.count() + node2.count())
            node1 = node1.next
            node2 = node2.next
        elif node1.doc_id > node2.doc_id:
            final_list[0].append(node2.doc_id)
            final_list[1].append(node2.count())
            node2 = node2.next
        else:
            final_list[0].append(node1.doc_id)
            final_list[1].append(node1.count())
            node1 = node1.next

    if not node1:
        while node2:
            final_list[0].append(node2.doc_id)
            final_list[1].append(node2.count())
            node2 = node2.next
    else:
        while node1:
            final_list[0].append(node1.doc_id)
            final_list[1].append(node1.count())
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
            final_list[1].append(node1.count() + node2.count())
            node1 = node1.next
            node2 = node2.next
        elif node1.doc_id > node2.doc_id:
            node2 = node2.next
        else:
            node1 = node1.next
    # print(final_list)
    # final_list[0], final_list[1] = [y for _, y in sorted(zip(final_list[1], final_list[0]), reverse=True)], \
    #                               sorted(final_list[1])

    # print(final_list)
    return arr_to_ll(final_list)


def not_and_docs(list1: mll.LinkedList, list2: mll.LinkedList) -> mll.LinkedList:
    """process `! AND`"""
    node1 = list1.head
    node2 = list2.head
    final_list = [[], []]
    while node1:
        if node1 and not node2 or node1.doc_id < node2.doc_id:
            final_list[0].append(node1.doc_id)
            final_list[1].append(node1.count())
            node1 = node1.next
        elif node1.doc_id == node2.doc_id:
            node1 = node1.next
            node2 = node2.next
        else:
            node2 = node2.next
    # final_list[0], final_list[1] = [y for _, y in sorted(zip(final_list[1], final_list[0]), reverse=True)], \
    #                                sorted(final_list[1])
    return arr_to_ll(final_list)


def get_expression_docs(words: list[str]):
    """return the documents which consist the expression"""
    ll_docs: mll.LinkedList = dictionary[words[0]]
    curr = ll_docs.head
    docs_counts, count, j = [[], []], 0, 0
    # print("##############")
    # print(words)
    # dictionary[words[0]].print_list()
    # print("##############")
    # dictionary[words[1]].print_list()
    # print("##############")

    while curr:
        for index in curr.indexes:
            # print(docs_tokens[curr.doc_id][index: index + len(words)])
            if docs_tokens[curr.doc_id][index: index + len(words)] == words:
                count += 1
        if count != 0:
            docs_counts[0].append(curr.doc_id)
            docs_counts[1].append(count)
        curr = curr.next
        count = 0
    # print(docs_counts)
    return arr_to_ll(docs_counts)


def process_query(query: str) -> tuple[set[str], list[int]]:
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
                    i += 1
                ll = get_expression_docs(words)
            else:
                if not dictionary[parsed_query[i]]:
                    continue
                ll = dictionary[parsed_query[i]]
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
            # ll.print_list()
            curr_res = and_docs(curr_res, ll)
            curr_res_backup = or_docs(curr_res_backup, ll)
            # curr_res.print_list()
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
    # curr_res_backup.print_list()
    final_sorted = [y for _, y in sorted(zip(final_list[1], final_list[0]), reverse=True)]
    print(f"normal: {final_sorted}")
    if curr_res.size < 5:  # in case we have less than 5 documents
        remove_repetitives(curr_res, curr_res_backup)
        final_list_backup = ll_to_arr(curr_res_backup)
        final_backup_sorted = [y for _, y in sorted(zip(final_list_backup[1], final_list_backup[0]),
                                                    reverse=True)]
        final_sorted.extend(final_backup_sorted)
        print(f"backup: {final_backup_sorted}")
    # print('final list: ')
    # print(final_list)
    return tokens, final_sorted


def remove_repetitives(res: mll.LinkedList, backup: mll.LinkedList):
    """merge the result of OR and AND"""
    res_curr = res.head
    backup_curr = backup.head
    while backup_curr and res_curr:
        # print("#################")
        # print(backup_curr.doc_id)
        # print(res_curr.doc_id)
        # print("#################")
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
    # print(y)
    while i in range(len(y)):
        if y[i][0] == "\"":
            z.append("$")  # '$' indicates start and end of the expression
            z.append(y[i][1:])
            i += 1
            while y[i][-1] != "\"":
                z.append(y[i])
                i += 1
            z.append(y[i][:-1])
            z.append("$")
        else:
            z.append(y[i])
        # print(f"z: {z}")
        i += 1
    # print(f"z[-1]: {z[-1]}")
    normalized = [normalizer.normalize(x) for x in z]
    # print(f"normalized: {normalized}")
    stemmed = [stemmer.stem(x) for x in normalized]
    # print(f"stemmer: {stemmed}")
    lemmatized = [lemmatizer.lemmatize(x) for x in stemmed]
    # print(f"lemmatized: {lemmatized}")
    final = [x for x in lemmatized if x != '']
    # print(f"final: {final}")
    return final


def retrieve_sentences(docs: list[int], tokens: set[str]) -> None:
    """print the sentences in which required words exist"""
    if len(docs) > 5:  # if there are many documents retrieve only first 5
        docs = docs[0:5]
    for doc_id in docs:
        title = data[str(doc_id)]['title']
        print(f'{bcolors.OKBLUE}document id: {doc_id}\ntitle: {title}\nsentences: ')
        sentences = sent_tokenize(data[str(doc_id)]['content'])
        for sent in sentences:
            normalized = normalizer.normalize(sent)
            tokenized = word_tokenize(normalized)
            stemmed = [stemmer.stem(x) for x in tokenized]
            lemmatized = [lemmatizer.lemmatize(x) for x in stemmed]
            if any(x in lemmatized for x in tokens):  # normal words expressions
                print(f"{bcolors.OKGREEN}{sent}")
            for token in tokens:
                if token[0] == "$":
                    words = token[1:].strip().split(" ")
                    # print(f"words: {words}'")
                    if is_subarray(lemmatized, words):
                        print(sent)


def take_snapshot():
    global dictionary
    global docs_tokens
    snapshot = Snapshot(_dictionary=dictionary, _docs_tokens=docs_tokens)
    snapshot_file = open("snapshot.obj", "wb")
    sys.setrecursionlimit(10000)
    pickle.Pickler(snapshot_file, protocol=pickle.HIGHEST_PROTOCOL).dump(pickle.dumps(snapshot))
    snapshot_file.close()


def restore_from_snapshot():
    global dictionary
    global docs_tokens
    snapshot_file = open("snapshot.obj", "rb")
    pickled = pickle.load(snapshot_file)
    snapshot: Snapshot = pickle.loads(pickled)
    dictionary = snapshot.dictionary
    docs_tokens = snapshot.docs_tokens
    snapshot_file.close()


def run():
    global dictionary
    global docs_tokens
    docs_tokens = build_tokens_lists(data)
    # for doc_tokens in docs_tokens:
    #     print(doc_tokens)
    dictionary = build_dict()
    # take_snapshot()
    # restore_from_snapshot()
    # for word in dictionary.keys():
    #     print(f'word: {word}\t\ttotal count: {dictionary[word].count})')
    #     dictionary[word].print_list()
    #     print("===========================")
    """main loop of the engine"""
    while True:
        query = input("پرسمان را وارد کنید:")
        # query = "\"خبر های ریاست\""
        print(query)
        tokens_and_docs = process_query(query)
        print(tokens_and_docs)
        retrieve_sentences(tokens_and_docs[1], tokens_and_docs[0])


dictionary = {}
docs_tokens = [[]]
run()

# print(docs_tokens)
# for x in docs_tokens:
#     print(x)
# print(dictionary.keys())
# words = ["خبر", "گزار", "آسیا"]
# ll = retrieve_docs(words[0])
# ll.print_list()
# tokens_and_docs = process_query("خبر ریاست")
# retrieve_sentences(tokens_and_docs[1], set(tokens_and_docs[0]))

# run()
