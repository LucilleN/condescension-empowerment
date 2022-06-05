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


def get_feature_vector(sentence, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category):
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

def load_or_generate_data(condescending_set, empowering_set, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category, abridged=False):
    data = [] 
    filename = "data.pkl"
    if abridged:
        filename = "data_abridged.pkl"
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

def plot_data(data):
    for column in data:
        plt.figure()
        plt.boxplot(data[column])
    plt.show()

def remove_outliers(data):
    # for col in data:
    #     for row in col:
    #         for elem in row:
    z_score = stattests.ztest(data)
    print(f"z_score: {z_score}")
    print(f"z_score size: {len(z_score)}")
    print(f"z_score[0] size: {len(z_score[0])}")
    print(f"z_score[1] size: {len(z_score[1])}")

    # rows_without_outliers = (np.abs(stats.zscore(df)) < 3).all(axis=1)
    # return data[rows_without_outliers]

if __name__ == "__main__":

    ### Read TalkDown data as condescending set
    condescending_set = read_talkdown()
    ### Read filtered Reddit scrape as empowering set
    empowering_set = read_filtered_reddit()
    empowering_set_abridged = read_filtered_reddit(abridged=True)

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

    data = load_or_generate_data(condescending_set, empowering_set, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category)

    data_description = descriptivestats.describe(data)
    data_description.to_csv('descriptive_stats.csv', sep='\t')

    for column in data:
        data_description = descriptivestats.describe(data[column])
        print(data_description)

    remove_outliers(data)
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