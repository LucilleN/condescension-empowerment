from utils import (
    read_talkdown, 
    read_filtered_reddit, 
    read_VAD_scores, 
    read_concreteness, 
    get_sentence_lexicon_score, 
    read_LIWC_lexicon, 
    get_LIWC_count,
    save_descriptive_stats)
import statsmodels.formula.api as smf
import statsmodels.stats.weightstats as stattests
import pandas as pd
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
from scipy import stats
import math


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

def load_or_generate_dataframe(condescending_set, empowering_set, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category, abridged=False):
    data = [] 
    filename = "data_abridged.pkl" if abridged else "data_unabridged.pkl"
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
    
    print(f"logistic_regression > load_or_generate_dataframe > len(data): {len(data)}")
    return data

def plot_data_boxplots(data, fig_title, subplot_names=None, num_rows=4, num_cols=5):

    fig, axs = plt.subplots(num_rows, num_cols)
    if subplot_names is None:
        subplot_names = data[columns]

    for index, column_name in enumerate(subplot_names):
        if column_name not in subplot_names:
            continue

        axs_y = int(math.ceil(index / num_cols)) - 1
        axs_x = index % num_cols - 1

        axs[axs_y, axs_x].boxplot(data[column_name])
        axs[axs_y, axs_x].set_title(column_name)

        # by default, 20 subplots in 4 rows, 5 cols
        # col   
        # 1     1  2  3  4  5
        # 2     6  7  8  9  10
        # 3     11 12 13 14 15
        # 4     16 17 18 19 20

    fig.subplots_adjust(bottom=0.05, top=0.9,
                        hspace=0.5, wspace=0.5)

    plt.suptitle(fig_title)

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

def print_model_summaries(models):
    for model_name, model in models.items():
        print(f"\n\n######### {model_name} #########\n")
        print(model.summary())


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

    save_descriptive_stats(data, 'descriptive_stats/unabridged.csv')
    save_descriptive_stats(data_abridged, 'descriptive_stats/abridged.csv')

    data_no_outliers = remove_outliers(data)
    data_abridged_no_outliers = remove_outliers(data_abridged)
    save_descriptive_stats(data_no_outliers, 'descriptive_stats/unabridged_no_outliers.csv')
    save_descriptive_stats(data_abridged_no_outliers, 'descriptive_stats/abridged_no_outliers.csv')

    different_datasets = {
        'Unabridged Data': data,
        'Abridged Data': data_abridged,
        'Unabridged Data w/o Outliers': data_no_outliers,
        'Abridged Data w/o Outliers': data_abridged_no_outliers
    }

    for dataset_name, dataset in different_datasets.items():
        # Plot a few selected columns
        plot_data_boxplots(
            dataset, 
            dataset_name, 
            subplot_names=['is_empowering', 'power', 'agency', 'sentiment', 'concreteness', "anger_count", "social_count", "relig_count", "sexual_count","humans_count"], 
            num_rows=3,
            num_cols=3
        )

    ### UNCOMMENT EVERYTHING BELOW
    models = {}

    lr_model_1 = smf.logit("is_empowering ~ power + agency + sentiment + concreteness", data=data).fit()
    models['VAD, CONCRETENESS, NO INTERACTIONS, FULL DATA'] = lr_model_1
    lr_model_2 = smf.logit("is_empowering ~ power + agency + sentiment + concreteness", data=data_abridged_no_outliers).fit()
    models['VAD, CONCRETENESS, NO INTERACTIONS, ABRIDGED DATA NO OUTLIERS'] = lr_model_2
    lr_model_3 = smf.logit("is_empowering ~ power * agency * sentiment * concreteness", data=data).fit()
    models['VAD, CONCRETENESS, WITH INTERACTIONS, FULL DATA'] = lr_model_3
    lr_model_4 = smf.logit("is_empowering ~ power + agency + sentiment + concreteness + anger_count + social_count + relig_count + sexual_count + humans_count", data=data).fit()
    models['VAD, CONCRETENESS, AND LIWC COUNTS, NO INTERACTIONS, FULL DATA'] = lr_model_4
   
    # Model 5 fails to converge, tried a lot of things
    # lr_model_5 = smf.logit("is_empowering ~ power * agency * sentiment * concreteness * anger_count * social_count * relig_count * sexual_count * humans_count", data=data).fit()
    # lr_model_5 = smf.logit("is_empowering ~ power * agency * sentiment * concreteness * anger_binary * social_binary * relig_binary * sexual_binary * humans_binary", data=data).fit()
    # Trying to exclude anger because it was not a significant predictor, as well as social_count and sexual_count because they were significant but had small beta's
    # Still doesn't converge for all of the below
    # lr_model_5 = smf.logit("is_empowering ~ power * agency * sentiment * concreteness * relig_count * humans_count", data=data).fit()
    # lr_model_5 = smf.logit("is_empowering ~ power * agency * sentiment * concreteness * relig_count * humans_count", data=data).fit(maxiter=100)
    # lr_model_5 = smf.logit("is_empowering ~ power * agency * sentiment * concreteness * humans_count", data=data).fit(maxiter=100)

    print_model_summaries(models)

    # print("\n\n######### MODEL 1: VAD, CONCRETENESS, NO INTERACTIONS, FULL DATA #########\n")
    # print(lr_model_1.summary())

    # print("\n\n######### MODEL 2: VAD, CONCRETENESS, NO INTERACTIONS, ABRIDGED DATA NO OUTLIERS #########\n")
    # print(lr_model_2.summary())

    # print("\n\n######### MODEL 3: VAD, CONCRETENESS, WITH INTERACTIONS, FULL DATA #########\n")
    # print(lr_model_3.summary())

    # print("\n\n######### MODEL 4: VAD, CONCRETENESS, AND LIWC COUNTS, NO INTERACTIONS, FULL DATA #########\n")
    # print(lr_model_4.summary())

    # This one fails to converge
    # print("\n\n######### MODEL 5: VAD, CONCRETENESS, AND LIWC COUNTS, WITH INTERACTIONS #########\n")
    # print(lr_model_4.summary())