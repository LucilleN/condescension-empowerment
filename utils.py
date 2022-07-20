from os.path import exists
import json
import pandas as pd
import numpy as np
import csv
import re
import statsmodels.stats.descriptivestats as descriptivestats
from scipy import stats
import matplotlib.pyplot as plt
import math
import seaborn as sns


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

def read_filtered_reddit(abridged=False, k=2602):
    """
    Reads the filtered data scraped from 8 empowering subreddits. 
    Returns:
        A list of strings, where each string is a post title
    """
    df = pd.read_csv("data/reddit_scrape_filtered.csv", sep="\t", header=None)
    empowering_set = df.values.reshape(-1).tolist() # reshape flattens it because every string is in its own list, making a big list of lists

    if abridged:
        # using power as a rough estimator of empowerment / condescension so that we can take only the top 2k
        empowering_set_with_power = get_sentences_with_power_scores(empowering_set)
        emp_sorted_by_power = sorted(empowering_set_with_power, key=lambda x: x['power'], reverse=True) 
        # Trim to length k
        emp_sorted_by_power_trimmed = emp_sorted_by_power[:k]
        sentences_only = [item['sentence'] for item in emp_sorted_by_power_trimmed]
        return sentences_only
    
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
        next(csv_reader)
        for row in csv_reader:
            # print(row)
            token = row[0]
            concreteness_mean_score = float(row[2])
            concreteness_scores[token] = concreteness_mean_score
    return concreteness_scores

def read_LIWC_lexicon():
    """
    Reads the LIWC lexicon.
    Returns:
        A dictionary, where the keys (str) are the names of categories, and values (list) are lists of words / word prefixes (str) that fall under that category.
    """
    liwc_words_by_category = {}
    category_num_to_name = {}

    with open(f'lexicons/LIWC2007_English080730.dic') as f:
        lines = f.readlines()
        words_start_at = None

        # read the categories first
        for index in range(1, len(lines)):
            line = lines[index]
            line = line.strip()
            if line == "%":
                words_start_at = index + 1
                break
            number, category = line.split("\t")
            # print(f"category: {category}, number: {number}")
            liwc_words_by_category[category] = []
            category_num_to_name[int(number)] = category

        # then read the lines with words followed by category numbers
        for line in lines[words_start_at:]:
            line = line.strip()
            line = re.sub("<.*?>|[()/ ]", "\t", line)
            elements = line.split("\t")
            word = elements[0]
            categories = elements[1:]
            for category_number in categories:
                if category_number == "":
                    continue
                category_name = category_num_to_name[int(category_number)]
                liwc_words_by_category[category_name].append(word)

    return liwc_words_by_category

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
            individual_word_scores.append(lexicon[word])
    # should I just skip anything that doesn't have any token in the power lexicon?
    if len(individual_word_scores) == 0: 
        return 0
    sentence_avg_power = sum(individual_word_scores) / len(individual_word_scores) # if len(individual_word_scores) > 0 else None
    return sentence_avg_power

def is_prefix(word, prefix):
    prefix = prefix[1:]
    return word.startswith(prefix)

def get_LIWC_count(sentence, liwc_words_by_category, category):
    """
    Goes through each token in a sentence, looks it up in the LIWC dictionary, and returns a list of counts of the specified categories
    Args:
        sentence: a string
        liwc_words_by_category: A dictionary, where the keys (str) are the names of categories, and values (list) are lists of words / word prefixes (str) that fall under that category
        category: the LIWC category to for which we're trying to count words in the sentence
    Returns:
        A tuple of three numbers (integers and float) that contains the count of words in the sentence that fall under the specified category, followed by its binary version (1 if count is anything greater than 0), followed by the count divided by the sentence length.
            Tuple format: (category_count, category_binary, category_normalized)
    """
    count = 0
    words_in_this_category = liwc_words_by_category[category]
    for word_in_category in words_in_this_category:
        # handle regular words
        if word_in_category in sentence.split():
            count = count + sentence.count(word_in_category)
        # handle word prefixes
        elif '*' in word_in_category:            
            for sentence_token in sentence.split():
                if is_prefix(word=sentence_token, prefix=word_in_category):
                    count = count + 1

    return (
        count,
        1 if count > 0 else 0,
        count / len(sentence))

def get_sentences_with_power_scores(sentences):
    sentences_with_power = []
    power_scores = read_VAD_scores('d')
    for sentence in sentences:
        sentence_avg_power = get_sentence_lexicon_score(sentence, power_scores)
        if sentence_avg_power is None:
            continue
        sentences_with_power.append({'sentence': sentence, 'power': sentence_avg_power})

    return sentences_with_power

def save_descriptive_stats(data, out_file):
    data_description = descriptivestats.describe(data)
    data_description.to_csv(out_file, sep='\t')

def print_model_summaries(models):
    for model_name, model in models.items():
        print(f"\n\n######### {model_name} #########\n")
        print(model.summary())
    
def remove_outliers(data):
    # print(f"data.size() is {np.size(data)}")
    print(f"data length is {len(data)}")
    rows_without_outliers = (np.abs(stats.zscore(data)) < 3).all(axis=1)
    trimmed_data = data[rows_without_outliers]
    print("AFTER REMOVING OUTLIERS")
    # print(f"data.size() is {np.size(trimmed_data)}")
    print(f"trimmed_data length is {len(trimmed_data)}")
    return trimmed_data


def get_feature_vector(sentence, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category):
    avg_power = get_sentence_lexicon_score(sentence, power_scores)
    avg_agency = get_sentence_lexicon_score(sentence, agency_scores)
    avg_sentiment = get_sentence_lexicon_score(sentence, sentiment_scores)

    avg_concreteness = get_sentence_lexicon_score(sentence, concreteness_scores)

    anger_count, anger_binary, anger_normalized = get_LIWC_count(sentence, liwc_words_by_category, "anger")
    social_count, social_binary, social_normalized = get_LIWC_count(sentence, liwc_words_by_category, "social")
    relig_count, relig_binary, relig_normalized = get_LIWC_count(sentence, liwc_words_by_category, "relig")
    sexual_count, sexual_binary, sexual_normalized = get_LIWC_count(sentence, liwc_words_by_category, "sexual")
    humans_count, humans_binary, humans_normalized = get_LIWC_count(sentence, liwc_words_by_category, "humans")

    feature_vector = [
        avg_power, 
        avg_agency, 
        avg_sentiment, 
        avg_concreteness,
        anger_count, anger_binary, anger_normalized,
        social_count, social_binary, social_normalized,
        relig_count, relig_binary, relig_normalized,
        sexual_count, sexual_binary, sexual_normalized,
        humans_count, humans_binary, humans_normalized]
    
    return feature_vector

def load_or_generate_dataframe(condescending_set, empowering_set, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category, abridged=False, matched=False):
    data = [] 
    filename = "data_abridged.pkl" if abridged else "data_unabridged.pkl"
    if matched:
        filename = "data_matched.pkl" 
    if exists(filename):
        print("loading data...")
        data = pd.read_pickle(filename)
    else:
        for sentence in condescending_set:
            data_point = [0] + get_feature_vector(sentence, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category)
            data.append(data_point)

        for sentence in empowering_set:
            data_point = [1] + get_feature_vector(sentence, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category)
            data.append(data_point)

        data = pd.DataFrame(data, columns = [
            'is_empowering', # the label: 0 means condescending, 1 means empowering
            'power', 
            'agency',
            'sentiment',
            'concreteness',
            "anger_count", "anger_binary", "anger_normalized",
            "social_count", "social_binary", "social_normalized",
            "relig_count", "relig_binary", "relig_normalized",
            "sexual_count", "sexual_binary", "sexual_normalized",
            "humans_count", "humans_binary", "humans_normalized"])

        print("saving data to file...")
        data.to_pickle(filename)
    
    return data

def plot_data(plot_type, data, fig_title, subplot_names=None, num_rows=4, num_cols=5):
    assert plot_type == "boxplot" or plot_type == "pdf"
    fig, axs = plt.subplots(num_rows, num_cols)
    
    if subplot_names is None:
        subplot_names = data[columns]

    for index, column_name in enumerate(subplot_names):
        if column_name not in subplot_names:
            continue

        axs_y = int(math.ceil(index / num_cols)) - 1
        axs_x = index % num_cols - 1

        if plot_type == "boxplot":
            axs[axs_y, axs_x].boxplot(data[column_name])
            axs[axs_y, axs_x].set_title(column_name)

        elif plot_type == "pdf":
            sns.histplot(
                data[column_name], 
                ax=axs[axs_y, axs_x], 
                # hist=True, 
                kde=True, 
                bins=int(20)
            )
                
            # axs[axs_x, axs_y].set_title(fig_title)
    
    fig.subplots_adjust(bottom=0.08, top=0.9, hspace=0.3, wspace=0.3)

    fig.suptitle(fig_title)

    plt.show()