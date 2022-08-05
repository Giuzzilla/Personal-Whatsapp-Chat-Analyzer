import re
from typing import Optional

import pandas as pd
import markovify
import nltk


def parse_chat(chat_path: str) -> pd.DataFrame:
    with open(chat_path, "r") as file:
        lines = list(file)

    messages = []
    i = 0

    while i < len(lines):
        lines[i] = lines[i].replace("\u200e", "")
        if bool(re.match(r"(.+/.+/.+), (..:..) - (.+):", lines[i])):  # CORRECT MESSAGE
            messages.append(lines[i].strip())
        elif bool(
            re.match(r".+/.+/.+, (..:..)", lines[i])
        ):  # SYSTEM MESSAGE, ONLY DATE PRESENT
            pass
        else:
            messages[-1] = (
                messages[-1] + " " + lines[i].strip()
            )  # CONTINUATION OF MESSAGE
        i += 1

    def split_string(s: str):
        matched = re.match(r"(.+/.+/.+, ..:..) - (.+): ([\s\S]+)", s)
        if not matched:
            raise ValueError(f"Invalid string which can't be splitted: {s}")
        timestamp = matched.group(1)
        author = matched.group(2)
        message = matched.group(3)
        return [timestamp, author, message]

    df = pd.DataFrame(columns=["timestamp", "author", "message"])
    message_series = pd.Series(messages)

    df["timestamp"] = message_series.transform(lambda s: split_string(s)[0])
    df["author"] = message_series.transform(lambda s: split_string(s)[1])
    df["message"] = message_series.transform(lambda s: split_string(s)[2])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


def avg_time_between(df: pd.DataFrame) -> float:
    df = df.loc[df["author"].shift() != df["author"]]

    return pd.Series(df.timestamp - df.shift().timestamp).mean()


def word_frequencies(df: pd.DataFrame, stopword_filter: bool = False):
    counts = df.message.str.lower().str.split(expand=True).stack().value_counts()
    counts = dict(counts)

    to_delete = ["<media", "omitted>"]

    if stopword_filter:
        nltk.download("stopwords")
        english_stopwords = nltk.corpus.stopwords.words("english")
        italian_stopwords = nltk.corpus.stopwords.words("italian")

        for word in counts.keys():
            if word in english_stopwords or word in italian_stopwords:
                to_delete.append(word)

    for word in to_delete:
        del counts[word]

    return counts


def generate_model(df: pd.DataFrame, author: Optional[str] = None):
    if author:
        df = df[df.author == author]
    return markovify.Text(df.message, state_size=2)


def generate_phrase_from_model(model):
    while True:
        sentence = model.make_sentence()
        if sentence is not None:
            break
    return sentence


def generate_phrase(df: pd.DataFrame, author: Optional[str] = None):
    model = generate_model(df, author)
    return generate_phrase_from_model(model)


if __name__ == "__main__":
    import sys

    parsed_df = parse_chat(sys.argv[1])
    freq = list(word_frequencies(parsed_df, stopword_filter=True).items())[:20]
    print("First 20 words by frequency: ")
    for i, word in enumerate(freq):
        print(f"{i + 1}°: '{word[0]}' - {word[1]}")

    print("\nAverage Delay between messages")
    print(avg_time_between(parsed_df))

    print("\nA random phrase generated from the whole chat: ")
    print(generate_phrase(parsed_df))
