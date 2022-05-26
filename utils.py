import json
import pandas as pd
import csv


def read_talkdown():
    """
    Reads condescending data from TalkDown.
    Returns:
        A list of strings, where each string is the 'quotedpost' field 
    """
    condescending_set = []
    with open("data/talkdown/data/balanced_train.jsonl") as f:
        json_list = list(f)
    for json_str in json_list:
        data = json.loads(json_str)
        if data['label']: # if it's labeled as True for condescending
            condescending_set.append(data['quotedpost']) # This is just the part that someone replied to pointing out that it's condescending
            # condescending_set.append(data['post']) # This is the entire post, which contains quotedpost plus more context
    return condescending_set

def read_filtered_reddit():
    """
    Reads the filtered data scraped from 8 empowering subreddits. 
    Returns:
        A list of strings, where each string is a post title
    """
    df = pd.read_csv("data/reddit_scrape_filtered.csv", sep="\t", header=None)
    # print(f"df size: {df.size}")
    empowering_set = df.values.reshape(-1).tolist() # reshape flattens it because every string is in its own list, making a big list of lists
    # print(empowering_set[:10])
    return empowering_set

def read_veiled_toxicity_clean():
    """
    Reads the unambiguously clean data from Han's Fortifying Toxic Speech Detectors paper 
    Returns:
        A list of strings
    """
    clean_set = pd.read_pickle('data/veiled-toxicity-detection/resources/processed_dataset/clean_train.pkl')
    # This is a list of tuples where the second element is always None, so 
    # just flattening it to a list of the strings
    clean_set = [sentence for sentence, _ in clean_set]
    return clean_set

def read_VAD_scores(dimension_to_load):
    """
    Reads dominance (aka power) from the VAD lexicon. 
    Args:
        dimension_to_load: a string with just 3 options to specify which dimension to load: 
                           "v" (valence / sentiment), "a" (arousal / agency), "d" (dominance / power)
    Returns:
        A dictionary, where keys (str) are the words in the lexicon and values (float) are their power score
    """
    assert dimension_to_load == "v" or dimension_to_load == "a" or dimension_to_load == "d"
    power_scores = {}
    with open(f'lexicons/NRC-VAD-Lexicon-Aug2018Release/OneFilePerDimension/{dimension_to_load}-scores.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            word, score = line.split("\t")
            power_scores[word] = float(score)
            # print(f"power_scores[{word}] = {float(score)}")
    return power_scores

def read_concreteness():
    """
    Reads concreteness scores from the concreteness lexicon.
    Returns: 
        A dictionary, where the keys (str) are the words in the lexicon and values (float) are their power score
    """
    # TODO: this currently doesn't account for the bigrams and only treats everything in the lexicon as a single word
    concreteness_scores = {}
    with open('lexicons/concreteness.csv', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            # print(row)
            token = row[0]
            concreteness_mean_score = row[2]
            concreteness_scores[token] = concreteness_mean_score
    print(concreteness_scores)
    return concreteness_scores


def get_sentence_lexicon_score(sentence, lexicon):
    """
    Goes through each token in a sentence, looks it up in a given lexicon, collects scores of all the words found, and returns their average 
    Args:
        sentence: a string 
        lexicon: a dictionary where the keys (str) are the words in the lexicon and values (float) are their score
    Returns:
        Average score of words in the sentence that did exist in the lexicon
        None if no words in the sentence were found in the lexicon
    """
    individual_word_scores = []
    for word in sentence.split():
        if word in lexicon:
            # print("found a word in the power scores")
            individual_word_scores.append(lexicon[word])
    # should I just skip anything that doesn't have any token in the power lexicon?
    if len(individual_word_scores) == 0: 
        return None
    sentence_avg_power = sum(individual_word_scores) / len(individual_word_scores) # if len(individual_word_scores) > 0 else None
    return sentence_avg_power