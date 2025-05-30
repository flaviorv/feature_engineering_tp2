from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

text = "A alma é, pois, imortal; renasceu repetidas vezes na existência e contemplou todas as coisas existentes e por isso não há nada que ela não conheça! Não é de espantar que ela seja capaz de evocar à memória a lembrança de objetos que viu anteriormente, e que se relacionam tanto com a virtude como com as outras coisas existentes. Toda a natureza, com efeito, é uma só, é um todo orgânico, e o espírito já viu todas as coisas; logo, nada impede que ao nos lembrarmos de uma coisa – o que nós, homens, chamamos de “saber” – todas as outras coisas acorram imediata e maquinalmente à nossa consciência."
text2 = "She walked through the silent woods, whispered to the ancient trees, and dreamed of days gone by."
tokens = word_tokenize(text)
tokens2 = word_tokenize(text2)
tagged = pos_tag(tokens)
tagged2 = pos_tag(tokens2)
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in tagged]
lemmatized_eng_words = [lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in tagged2]

print(lemmatized_words)
print(lemmatized_eng_words)