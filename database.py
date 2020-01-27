import requests
from bs4 import BeautifulSoup


def save_html(html, path):
    with open(path, 'wb') as f:
        f.write(html)


def open_html(path):
    with open("html/"+path, 'rb') as f:
        return f.read()


# https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists#Korean
# http://frequencylists.blogspot.com/2015/12/the-2000-most-frequently-used-korean.html
# ^ maybe for korean
languages = {"english": ['https://en.wiktionary.org/wiki/Appendix:1000_basic_English_words',
                    'https://simple.wiktionary.org/wiki/Wiktionary:Most_frequent_1000_words_in_English',
                    'https://en.wiktionary.org/wiki/Appendix:Basic_English_word_list'],
                    "dutch": ['https://en.wiktionary.org/wiki/Appendix:1000_basic_Dutch_words',
                              'https://en.wiktionary.org/wiki/Category:Dutch_compound_words',
                              'https://en.wiktionary.org/w/index.php?title=Category:Dutch_compound_words&from=A',
                              'https://en.wiktionary.org/w/index.php?title=Category:Dutch_compound_words&from=D',
                              'https://en.wiktionary.org/w/index.php?title=Category:Dutch_compound_words&from=F',
                              'https://en.wiktionary.org/w/index.php?title=Category:Dutch_compound_words&from=M',
                              'https://en.wiktionary.org/w/index.php?title=Category:Dutch_compound_words&from=P',
                              'https://en.wiktionary.org/w/index.php?title=Category:Dutch_compound_words&from=X'],
                    "japanese": ['https://en.wiktionary.org/wiki/Appendix:1000_Japanese_basic_words'],
                    "mandarin": ['https://en.wiktionary.org/wiki/Appendix:HSK_list_of_Mandarin_words/Beginning_Mandarin'],
                    "thai": ['https://en.wiktionary.org/wiki/Appendix:200_basic_Thai_words'],
                    "vietnamese": ['https://en.wiktionary.org/wiki/Appendix:1000_most_common_Vietnamese_words'],
                    "french": ['https://en.wiktionary.org/wiki/Category:French_basic_words',
                               'https://en.wiktionary.org/wiki/Category:French_3-syllable_words',
                               'https://en.wiktionary.org/wiki/Category:French_2-syllable_words',
                               'https://en.wiktionary.org/w/index.php?title=Category:French_2-syllable_words&from=J',
                               'https://en.wiktionary.org/w/index.php?title=Category:French_2-syllable_words&from=C'
                               'https://en.wiktionary.org/w/index.php?title=Category:French_4-syllable_words&from=Z'],
                    "ido": ['https://en.wiktionary.org/wiki/Appendix:1000_basic_Ido_words'],
                    "korean": ['https://en.wiktionary.org/wiki/Appendix:Common_Korean_words'],
                    "spanish": ['https://en.wiktionary.org/wiki/Category:Spanish_basic_words',
                                 'https://en.wiktionary.org/w/index.php?title=Category:Spanish_basic_words&pagefrom=limpio#mw-pages',
                                 'https://en.wiktionary.org/w/index.php?title=Category:Spanish_basic_words&pagefrom=vaca#mw-pages',
                                 'https://en.wiktionary.org/w/index.php?title=Category:Spanish_surnames&from=A',
                                 'https://en.wiktionary.org/w/index.php?title=Category:Spanish_surnames&from=D',
                                 'https://en.wiktionary.org/w/index.php?title=Category:Spanish_surnames&from=M',
                                 'https://en.wiktionary.org/w/index.php?title=Category:Spanish_surnames&from=S']}


list_of_vocabularies = {"english": [],
                    "dutch": [],
                    "japanese": [],
                    "mandarin": [],
                    "thai": [],
                    "vietnamese": [],
                    "spanish": [],
                    "french": [],
                    "ido": [],
                    "korean": []}


def save_languages(languages):
    for language in languages:
        links = len(languages[language])
        for link in range(links):
            url = languages[language][link]
            r = requests.get(url)
            save_html(r.content, "html/"+language+str(link))


def get_vocabularies():
    latin_languages = ["english", "dutch", "french", "ido", "spanish"]
    non_latin_languages = ["japanese", "mandarin", "thai", "korean", "vietnamese"]

    for language in latin_languages:
        links = len(languages[language])
        for link in range(links):
            try:
                html = open_html(language + str(link))
            except FileNotFoundError:
                save_languages(languages)
                html = open_html(language + str(link))
            soup = BeautifulSoup(html, 'html.parser')
            words = []
            for word in soup.find_all('a'):
                if word.get_text() == word.get('title'):
                    # edge cases
                    if word.get('title').__contains__("Category"):
                        continue
                    if word.get('title').__contains__(" "):
                        continue
                    #if already in language words list
                    if  word.get('title') in list_of_vocabularies[language]:
                        continue
                    # add to words
                    words.append(word.get('title'))
            # edge cases
            if language == "english":
                words = words[:-1]
            elif language == "dutch":
                words = words[1:]
            # add to existing dictionary for that language
            list_of_vocabularies[language].extend(words)

    # Non-Latin
    for language in non_latin_languages:
        links = len(languages[language])
        for link in range(links):
            html = open_html(language + str(link))
            soup = BeautifulSoup(html, 'html.parser')
            words = []
            if language == "japanese":
                text = soup.find_all('i')
                for line in text:
                    text = line.get_text()
                    if "," in text:
                        fixedText = text.split(",")
                        for element in fixedText:
                            if element != '':
                                words.append(element.strip())
                    else:
                        words.append(text)
            if language == "thai" or language == "korean":
                text = soup.find_all("span", class_="tr Latn")
                for line in text:
                    words.append(line.get_text())
            if language == "mandarin":
                text = soup.find_all("span", class_="tr Latn")
                for line in text:
                    text = line.get_text()
                    if "/" in text:
                        fixedText = text.split("/")
                        for element in fixedText:
                            if element != '':
                                words.append(element.strip())
                    elif "..." in text:
                        fixedText = text.split("...")
                        for element in fixedText:
                            if element != '':
                                words.append(element.strip())
                    else:
                        words.append(line.get_text())
            elif language == "vietnamese":
                text = soup.find_all('span', class_="Latn")
                for line in text:
                    words.append(line.get_text())
            list_of_vocabularies[language] = words
    return list_of_vocabularies

#
# if __name__ == "__main__":
#     get_vocabularies()
#
#     # FOR TESTING
#     x = get_vocabularies()
#
#     for language in x:
#         print(language, " size: {}".format(len(x[language])))
#
