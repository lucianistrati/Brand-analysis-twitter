import os
import ast
import pickle
import string
import re
import nltk
import copy
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from itertools import chain, combinations

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


class TweetPreprocessor:
    def __init__(
            self,
            stemmer=SnowballStemmer('romanian'),
            customs_sws=None,
            abbrev_dict=None,
            emoji_dict=None,
            cities_list=None,
            multiple_vowel_words_set=None,
            companies_list=None,
            allow_stemming=True):
        """
        Class used in for preprocessing the tweets
        Parameters
        ----------
        stemmer : stemmer object
            used to extract the roots of the words
        customs_sws : numpy array
            list of the stop-words used by the preprocessor
        abbrev_dict : numpy array
            dictionary that maps a string to another string
        emoji_dict : numpy array
            dictionary that maps an emoticon to labels "bun" or "rau"
        cities_list : numpy array
            list of strings
        multiple_vowel_words_set : dict
            set containing the words in Romanian that contain duplicated vowels
        """
        self.stemmer = stemmer
        self.custom_sws = customs_sws if customs_sws is not None else self.read_list_of_stop_words()
        self.abbrev_dict = abbrev_dict if abbrev_dict is not None else self.read_list_of_common_abbreviations()
        self.emoji_dict = emoji_dict if emoji_dict is not None else self.read_emoji_dictionary()
        self.cities_list = cities_list if cities_list is not None else self.read_list_of_cities()
        self.multiple_vowel_words_set = multiple_vowel_words_set if multiple_vowel_words_set is not None else self.read_list_of_words_with_multiple_vowels()
        self.companies_list = companies_list if companies_list is not None else self.read_list_of_companies()
        self.allow_stemming = allow_stemming

    def read_list_of_stop_words(self):
        path = ""
        print(str(os.getcwd()))
        if "Brand-analysis-master" not in str(os.getcwd()):
            path = "Brand-analysis-master/Data/Custom stop-words and emojis/custom_stop_words_list.npy"
        else:
            path = "Data/Custom stop-words and emojis/custom_stop_words_list.npy"
        custom_sws = np.load(path, allow_pickle=True)

        return custom_sws

    def read_list_of_words_with_multiple_vowels(self):
        path = ""
        if "Brand-analysis-master" not in str(os.getcwd()):
            path = "Brand-analysis-master/Data/Romanian words with their lemma's/multiple_vowel_words_rom_lang.txt"
        else:
            path = "Data/Romanian words with their lemma's/multiple_vowel_words_rom_lang.txt"
        multiple_vowel_words_set = set()
        with open(path, encoding='utf-8', mode="r") as f:
            lines = f.readlines()
            for line in lines:
                multiple_vowel_words_set.add(line[:-1])
        return multiple_vowel_words_set

    def read_list_of_common_abbreviations(self):
        path = ""
        if "Brand-analysis-master" not in str(os.getcwd()):
            path = "Brand-analysis-master/Data/Custom stop-words and emojis/common_abbreviations_ro.txt"
        else:
            path = "Data/Custom stop-words and emojis/common_abbreviations_ro.txt"

        abbrev_dict = {}
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                words = line.split("=")
                if len(words) == 1:
                    break
                words[0] = " " + words[0]
                words[1] = words[1][:-1] + " "
                abbrev_dict[words[0]] = words[1]
        return abbrev_dict

    def read_emoji_dictionary(self):
        path = ""
        if "Brand-analysis-master" not in str(os.getcwd()):
            path = "Brand-analysis-master/Data/Custom stop-words and emojis/emoji_dict.npy"
        else:
            path = "Data/Custom stop-words and emojis/emoji_dict.npy"

        emoji_dict = np.load(path, allow_pickle=True)
        return emoji_dict

    def read_list_of_cities(self):
        path = ""
        if "Brand-analysis-master" not in str(os.getcwd()):
            path = "Brand-analysis-master/Data/Custom stop-words and emojis/cities_list.npy"
        else:
            path = "Data/Custom stop-words and emojis/cities_list.npy"
        cities_list = np.load(path, allow_pickle=True)
        return cities_list

    def read_list_of_companies(self):
        path = ""
        if "Brand-analysis-master" not in str(os.getcwd()):
            path = "Brand-analysis-master/Data/Companies/companies_list.npy"
        else:
            path = "Data/Companies/companies_list.npy"
        global companies_list
        companies_list = np.load(path, allow_pickle=True)

        return companies_list

    def is_number(self, text):
        for elem in text:
            if elem < '0' or elem > '9':
                return False
        return True

    def is_diacritic(self, letter):
        for diacritic in ['ă', 'â', 'ș', 'ț', 'î']:
            if letter.lower() == diacritic:
                return True
        return False

    def eliminate_diacritics(self, word):
        word = word.replace("ă", "a")
        word = word.replace("â", "a")
        word = word.replace("î", "i")
        word = word.replace("ț", "t")
        word = word.replace("ș", "s")
        word = word.replace("Ă", "A")
        word = word.replace("Â", "A")
        word = word.replace("Î", "I")
        word = word.replace("Ț", "T")
        word = word.replace("Ș", "S")

        return word

    def remove_at(self, i, s):
        return s[:i] + s[i + 1:]

    def powerset(self, iterable):
        """
        Parameters
        ----------
        entity_name : string

        Returns
        ----------
        iterable : list
            the list contains tuples of integers (the powerset of the initial list)
        """
        s = list(iterable)

        return chain.from_iterable(
            combinations(
                s, r) for r in range(
                len(s) + 1))

    def has_extra_letters(self, text):
        """
        Parameters
        ----------
        text : string

        Returns
        ----------
        boolean : True if there are any two identical letters one next to other, False either
        """
        for i in range(len(text) - 1):
            if text[i] == text[i + 1]:
                return True

        return False

    def eliminate_extra_letters(self, word):
        """
        Parameters
        ----------
        word : string

        Returns
        ----------
        word : string
            the returned string is gramatically corect in romanian

        There are about 25.000 words in romanian language that contain two or more than two unified vowels.
        From these 25.000 words they contain the following groups of vowels "aa", "ee", "ii", "uu", "oo" or "iii"
        """
        if word in self.multiple_vowel_words_set:
            return word

        vowels = {'a', 'e', 'i', 'o', 'u'}
        i = 0
        while i < len(word[:-1]):
            if word[i] == word[i + 1]:
                if word[i] not in vowels:
                    word = self.remove_at(i, word)
                    i -= 1
            i += 1

        i = 0
        while i < len(word[:-2]):
            if word[i] == word[i + 1] and word[i + 1] == word[i + \
                2] and word[i] in vowels and word[i] != 'i':
                word = self.remove_at(i, word)
                i -= 1
            i += 1
        i = 0
        while i < len(word[:-3]):
            if word[i] == word[i + 1] and word[i + 1] == word[i + \
                2] and word[i + 2] == word[i + 3] and word[i] == 'i':
                word = self.remove_at(i, word)
                i -= 1
            i += 1
        if word in self.multiple_vowel_words_set:
            return word
        positions = [0] * len(word)
        counter = 1
        positions[0] = counter
        for i in range(1, len(word)):
            if word[i] == word[i - 1]:
                positions[i] = counter
            else:
                counter += 1
                positions[i] = counter
        trials = []
        for i in range(len(positions) - 1):
            if positions[i] == positions[i + 1]:
                trials.append(i + 1)
        all_delete_combinations = self.powerset(trials)
        first_copy = copy.deepcopy(word)
        for comb in all_delete_combinations:
            word = copy.deepcopy(first_copy)
            comb = sorted(comb, reverse=True)
            for x in comb:
                word = self.remove_at(x, word)
            if word in self.multiple_vowel_words_set:
                return word
        word = first_copy
        i = 0
        while i < len(word[:-1]):
            if word[i] == word[i + 1]:
                word = self.remove_at(i, word)
                i -= 1
            i += 1

        return word

    def compare_ignore_case(self, a, b):
        return a.lower() in b.lower()

    def is_camel_case(self, text):
        """
        Parameters
        ----------
        text : string

        Returns
        ----------
        bool : True/False
            the functions returns True if the string given as argument
            respects the CamelCase rules
        """
        capital_letters = 0
        text = re.sub('[\\W]+', '', text)
        for character in text:
            if (character < 'A' or character > 'Z') and (
                    character < 'a' or character > 'z'):
                return False
            if character >= 'A' and character <= 'Z':
                capital_letters += 1
        if capital_letters < 2:
            return False
        for i in range(len(text)):
            if text[i] >= 'A' and text[i] <= 'Z':
                if i == len(text) - 1:
                    return False
            elif i == 0:
                return False
        for i in range(len(text) - 2):
            if text[i] == text[i + 1] and text[i + 1] == text[i + \
                2] and text[i] >= 'A' and text[i] <= 'Z':
                return False
        return True

    def get_words_from_camel_case(self, text):
        """
        Parameters
        ----------
        text : string

        Returns
        ----------
        words : list
            the list contains the words that are extracted from the initial string
            when dividing merged words like "NuOSaCrezi" into the list ['Nu', 'o', 'Sa', 'Crezi']
        """
        words = []
        word = ""
        for i in range(len(text)):
            if text[i] >= 'A' and text[i] <= 'Z':
                if word != "":
                    words.append(word)
                word = ""
                word += text[i]
            else:
                word += text[i]
        words.append(word)

        return words

    def preprocess_tweet(self, text):
        """
        Parameters
        ----------
        text : string

        Returns
        ----------
        processed_list : list
            the list contains strings which are obtained after doing the necessary preprocessing before the labelling should take place
        """

        # remove url's
        text = re.sub(
            '((www\\.[^\\s]+)|(https?://[^\\s]+)|(http?://[^\\s]+))',
            '',
            text)
        # remove digits
        text = re.sub('\\d+', '', text)
        # remove url's
        text = re.sub(r'http\S+', '', text)
        # remove usernames
        text = re.sub('@[^\\s]+', '', text)
        # replacing # with space
        text = text.replace("#", ' ')
        tokens_list = text.split(" ")
        text = ""
        for i in range(len(tokens_list)):
            if " " + tokens_list[i].lower() + " " in self.abbrev_dict.keys():
                tokens_list[i] = self.abbrev_dict[" " +
                                                  tokens_list[i].lower() + " "]
            tokens_list[i] = self.eliminate_diacritics(tokens_list[i])
            is_company = False
            for comp in companies_list:
                if self.compare_ignore_case(comp, tokens_list[i]):
                    is_company = True
                    break
            if self.has_extra_letters(tokens_list[i].lower()):
                if not is_company:
                    tokens_list[i] = self.eliminate_extra_letters(
                        tokens_list[i].lower())
            if is_company == False and self.is_camel_case(
                    tokens_list[i]) == True:
                splitted_words = self.get_words_from_camel_case(tokens_list[i])
                for new_word in splitted_words:
                    if " " + new_word.lower() + " " in self.abbrev_dict.keys():
                        new_word = self.abbrev_dict[" " +
                                                    new_word.lower() + " "]
                    text += new_word
                    text += " "
            else:
                text += tokens_list[i]
                text += " "
            try:
                value = self.emoji_dict[()][tokens_list[i]]
                text = text.replace(tokens_list[i], value)
            except BaseException:
                continue
        # Eliminating any punctuation mark
        text = re.sub('[\\W]+', ' ', text)
        nopunc = [char for char in text]
        # Join the characters again to form the string.
        nopunc = ''.join(nopunc)
        # Tokenizing into a list of tokens
        nopunc = word_tokenize(nopunc)
        # Using Snowball Stemmer for Romanian Language in order to stem the
        # words and convert them to lowercase
        stemmed_list = []
        for token in nopunc:
            should_be_stemmed = True
            should_be_kept = False
            if token == 'bun' or token == 'rau':
                should_be_stemmed = False
                should_be_kept = True
            if should_be_stemmed:
                for city in self.cities_list:
                    if self.compare_ignore_case(token, city):
                        should_be_stemmed = False
                        break
            if should_be_stemmed:
                for comp in companies_list:
                    if self.compare_ignore_case(token, comp):
                        should_be_stemmed = False
                        break
            if should_be_stemmed:
                for sw in self.custom_sws:
                    if self.compare_ignore_case(token, sw):
                        should_be_stemmed = False
                        should_be_kept = True
                        break
            if should_be_stemmed:
                if not self.allow_stemming:
                    stemmed_list.append(token.lower())
                else:
                    stemmed_list.append(self.stemmer.stem(token.lower()))
            elif should_be_stemmed == False and should_be_kept == True:
                stemmed_list.append(token.lower())
            elif token.lower() == 'nu':
                stemmed_list.append('nu')

        replaced_list = []
        for i in range(len(stemmed_list)):
            if " " + stemmed_list[i] + " " in self.abbrev_dict.keys():
                replacement = self.abbrev_dict[" " + stemmed_list[i] + " "]
                words = [word.strip() for word in replacement.split()]
                for word in words:
                    replaced_list.append(word)
            else:
                replaced_list.append(stemmed_list[i])
        processed_list = [
            word for word in replaced_list if word not in self.custom_sws]

        return processed_list

    def read_utils(self):
        self.read_list_of_stop_words()
        self.read_list_of_words_with_multiple_vowels()
        self.read_list_of_common_abbreviations()
        self.read_emoji_dictionary()
        self.read_list_of_cities()
        self.read_list_of_companies()

    def preprocess_dataframe(self, dataframe):
        """
        Parameters
        ----------
        dataframe : pandas.DataFrame
            receives a dataframe for which we will going to process the "Tweet" column

        Returns
        ----------
        processed_list : list
            the list contains strings which are obtained after doing the necessary preprocessing
        """
        self.read_utils()
        for i in range(len(dataframe)):
            dataframe.iloc[i]['Tweet'] = self.preprocess_tweet(
                dataframe.iloc[i]['Tweet'])
        return dataframe
