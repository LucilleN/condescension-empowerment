import json
import pandas as pd


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