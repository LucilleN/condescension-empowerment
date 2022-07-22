from scrape_reddit import subreddits
import pandas as pd
import statistics
from utils import read_talkdown


dfs_for_each_subreddit = []
for subreddit in subreddits:
    df = pd.read_csv(f"data/reddit_scrape/{subreddit}.csv", sep='\t')
    # post_titles = list(df["title"])
    print(f"{subreddit} has {len(df)} posts")
    dfs_for_each_subreddit.append(df)
    # post_titles = list(post_titles)
    # print(post_titles[:10])
all_data = pd.concat(dfs_for_each_subreddit)
print(f"In total there are {len(all_data)} posts")


### Read condescending data from TalkDown
condescending_set = read_talkdown()
sentence_lengths = [len(sentence) for sentence in condescending_set]
mean_length = statistics.mean(sentence_lengths)
sd_length = statistics.stdev(sentence_lengths)
print(f"Mean of TalkDown length is {mean_length}")
print(f"Standard Deviation of TalkDown length is {sd_length}")

min_length = int(mean_length) - int(sd_length)
max_length = int(mean_length) + int(sd_length)

def contains_2nd_person_pronouns(tokens):
    pronouns = [
        "you",
        "your",
        "youre",
        "you're",
        "yall",
        "y'all",
        "u",
        "ur"
    ]
    for pronoun in pronouns:
        if pronoun in tokens:
            return True
    return False


# print(f"all_data size: {all_data.size}")
# print("all_data:")
# print(all_data.to_string())

filtered_data = []
for index, row in all_data.iterrows():
    sentence = row[0]
    score = row[1]
    print("iterating through all_data")
    print(f"printing sentence: {sentence}")

    if not type(sentence) == str:
        print("skipping this sample because it's not a string")
        continue

    # TEMPORARY -- USE A MORE INTELLIGENT TOKENIZER!
    tokens = sentence.lower().split(" ")
    print(f"tokens: {tokens}")
    
    # filter by length: find the mean and variance/sd of talkdown, then only keep posts that have length of mean +/- 1 or 2 sd. or 1.5 sd?? idk
    if len(tokens) < min_length or len(tokens) > max_length:
        continue
    
    # filter for words that have "you" and "your"
    if not contains_2nd_person_pronouns(tokens):
        continue
    
    filtered_data.append((sentence, score))

print(f"After filtering there are {len(filtered_data)} posts")
# After filtering there are 208000 posts

df = pd.DataFrame(filtered_data, columns=["sentence", "score"])
df.to_csv("data/reddit_scrape_filtered_with_metadata.csv", sep="\t", index=False, header=False)