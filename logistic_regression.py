from utils import (
    read_talkdown, 
    read_filtered_reddit, 
    read_VAD_scores, 
    read_concreteness, 
    get_sentence_lexicon_score, 
    read_LIWC_lexicon, 
    get_LIWC_count)
import statsmodels.formula.api as smf
import statsmodels.stats.weightstats as stattests
import statsmodels.stats.descriptivestats as descriptivestats
import pandas as pd
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
from scipy import stats


def get_feature_vector(sentence, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category):
    # print(f"in logistic_regression > get_feature_vector > sentence is type {type(sentence)}: {sentence}")
    # print("Extracting VAD features...")
    avg_power = get_sentence_lexicon_score(sentence, power_scores)
    avg_agency = get_sentence_lexicon_score(sentence, agency_scores)
    avg_sentiment = get_sentence_lexicon_score(sentence, sentiment_scores)

    # print("Extracting Concreteness features...")
    avg_concreteness = get_sentence_lexicon_score(sentence, concreteness_scores)

    # print("Extracting LIWC features...")
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
    
    # print(feature_vector)
    return feature_vector

def load_or_generate_dataframe(condescending_set, empowering_set, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category, abridged=False):
    data = [] 
    filename = "data_abridged.pkl" if abridged else "data_unabridged.pkl"
    # if abridged:
    #     filename = "data_abridged.pkl"
    if exists(filename):
        print("loading data...")
        data = pd.read_pickle(filename)
    else:
        for sentence in condescending_set:
            # print("SENTENCE")
            # print(sentence)
            # print(str(sentence))
            # sentence = str(sentence)
            # print(type(sentence))
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
    
    print(f"logistic_regression > load_or_generate_dataframe > len(data): {len(data)}")
    return data

def plot_data(data):
    for column in data:
        plt.figure()
        plt.boxplot(data[column])
    plt.show()

def remove_outliers(data):
    # print(f"data.size() is {np.size(data)}")
    print(f"data length is {len(data)}")
    rows_without_outliers = (np.abs(stats.zscore(data)) < 3).all(axis=1)
    trimmed_data = data[rows_without_outliers]
    print("AFTER REMOVING OUTLIERS")
    # print(f"data.size() is {np.size(trimmed_data)}")
    print(f"trimmed_data length is {len(trimmed_data)}")
    return trimmed_data

def save_descriptive_stats(data, out_file):
    data_description = descriptivestats.describe(data)
    data_description.to_csv(out_file, sep='\t')

if __name__ == "__main__":

    ### Read TalkDown data as condescending set
    condescending_set = read_talkdown()
    print(f"len(condescending_set): {len(condescending_set)}")
    ### Read filtered Reddit scrape as empowering set
    empowering_set = read_filtered_reddit()
    empowering_set_abridged = read_filtered_reddit(abridged=True, k=len(condescending_set))
    print(f"len(empowering_set): {len(empowering_set)}")
    print(f"len(empowering_set_abridged): {len(empowering_set_abridged)}")


    ### Load VAD lexicon
    sentiment_scores = read_VAD_scores("v") # v for valence
    agency_scores = read_VAD_scores("a") # a for agency
    power_scores = read_VAD_scores("d") # d for dominance

    ### Load concreteness lexicon
    concreteness_scores = read_concreteness()

    ### Load LIWC
    liwc_words_by_category = read_LIWC_lexicon()

    ### Feature extraction
    X = [] # a list of lists, where rows are samples and columns are features
    y = [] # a list of 0's and 1's corresponding to the label of each sample. 0 = condescension, 1 = empowerment

    data = load_or_generate_dataframe(condescending_set, empowering_set, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category)
    data_abridged = load_or_generate_dataframe(condescending_set, empowering_set_abridged, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category, abridged=True)

    save_descriptive_stats(data, 'descriptive_stats_unabridged.csv')
    save_descriptive_stats(data_abridged, 'descriptive_stats_abridged.csv')

    data_no_outliers = remove_outliers(data)
    data_abridged_no_outliers = remove_outliers(data_abridged)
    save_descriptive_stats(data_no_outliers, 'descriptive_stats_unabridged_no_outliers.csv')
    save_descriptive_stats(data_abridged_no_outliers, 'descriptive_stats_abridged_no_outliers.csv')

    different_datasets = [
        data,
        data_abridged,
        data_no_outliers,
        data_abridged_no_outliers
    ]


    # plot_data(data)

    ### UNCOMMENT EVERYTHING BELOW


    # lr_model_1 = smf.logit("is_empowering ~ power + agency + sentiment + concreteness", data=data).fit()
    
    # lr_model_2 = smf.logit("is_empowering ~ power * agency * sentiment * concreteness", data=data).fit()
    
    # lr_model_3 = smf.logit("is_empowering ~ power + agency + sentiment + concreteness + anger_count + social_count + relig_count + sexual_count + humans_count", data=data).fit()
   
    # # This one fails to converge
    # # lr_model_4 = smf.logit("is_empowering ~ power * agency * sentiment * concreteness * anger_count * social_count * relig_count * sexual_count * humans_count", data=data).fit()
    # # lr_model_4 = smf.logit("is_empowering ~ power * agency * sentiment * concreteness * anger_binary * social_binary * relig_binary * sexual_binary * humans_binary", data=data).fit()
    
    # # Trying to exclude anger because it was not a significant predictor, as well as social_count and sexual_count because they were significant but had small beta's
    # # Still doesn't converge for all of the below
    # # lr_model_4 = smf.logit("is_empowering ~ power * agency * sentiment * concreteness * relig_count * humans_count", data=data).fit()
    # # lr_model_4 = smf.logit("is_empowering ~ power * agency * sentiment * concreteness * relig_count * humans_count", data=data).fit(maxiter=100)
    # # lr_model_4 = smf.logit("is_empowering ~ power * agency * sentiment * concreteness * humans_count", data=data).fit(maxiter=100)

    # print("\n\n######### MODEL 1: VAD, CONCRETENESS, NO INTERACTIONS #########\n")
    # print(lr_model_1.summary())

    # print("\n\n######### MODEL 2: VAD, CONCRETENESS, WITH INTERACTIONS #########\n")
    # print(lr_model_2.summary())

    # print("\n\n######### MODEL 3: VAD, CONCRETENESS, AND LIWC COUNTS, NO INTERACTIONS #########\n")
    # print(lr_model_3.summary())

    # # This one fails to converge
    # # print("\n\n######### MODEL 4: VAD, CONCRETENESS, AND LIWC COUNTS, WITH INTERACTIONS #########\n")
    # # print(lr_model_4.summary())