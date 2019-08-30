import pandas as pd
import re
import markovify
import nltk 

def parseChat(chatPath):
    with open(chatPath, 'r') as file:
        lines = list(file)

    messages = []
    i = 0

    while i < len(lines):
        if bool(re.match(r'(.+/.+/.+), (..:..) - (.+):', lines[i])): # CORRECT MESSAGE
            messages.append(lines[i].strip())
        elif bool(re.match(r'.+/.+/.+, (..:..)', lines[i])): # SYSTEM MESSAGE, ONLY DATE PRESENT
            pass
        else:
            messages[-1] = messages[-1] + " " + lines[i].strip() # CONTINUATION OF MESSAGE
        i += 1

    def splitS(s): 
        regex = re.match(r'(.+/.+/.+, ..:..) - (.+): ([\s\S]+)', s)
        timestamp = regex.group(1)
        author = regex.group(2)
        message = regex.group(3)
        return [timestamp, author, message]

    df = pd.DataFrame(columns = ['timestamp', 'author', 'message'])
    messageSeries = pd.Series(messages)

    df['timestamp'] = messageSeries.transform(lambda s: splitS(s)[0])
    df['author'] = messageSeries.transform(lambda s: splitS(s)[1])
    df['message'] = messageSeries.transform(lambda s: splitS(s)[2])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df

def avgTimeBtwn(df):
    df = df.loc[df['author'].shift() != df['author']]
    
    return pd.Series(df.timestamp - df.shift().timestamp).mean()


def wordFrequencies(df, stopwordFilter = False):
    counts = df.message.str.lower().str.split(expand=True).stack().value_counts()
    counts = dict(counts)

    to_delete = ['<media', 'omitted>']

    if stopwordFilter:
        nltk.download('stopwords')
        english_stopwords = nltk.corpus.stopwords.words('english')
        italian_stopwords = nltk.corpus.stopwords.words('italian')

        for word in counts.keys():
            if word in english_stopwords or word in italian_stopwords: 
                to_delete.append(word)
               
    for word in to_delete:
        del counts[word]

    return counts


def generatePhrase(df, author = None):
    if author:
        df = df[df.author == author]

    model = markovify.NewlineText(df.message, state_size = 2)
    while True:    
        sentence = model.make_sentence()
        if sentence != None:
            break
    return sentence


if __name__ == '__main__':
    import sys
    df = parseChat(sys.argv[1])
    freq = list(wordFrequencies(df, stopwordFilter = True).items())[:20]
    print("First 20 words by frequency: ")
    for i, word in enumerate(freq):
        print(f"{i + 1}°: '{word[0]}' - {word[1]}")

    print("\nAverage Delay between messages")
    print(avgTimeBtwn(df))

    print("\nA random phrase generated from the whole chat: ")
    print(generatePhrase(df))

