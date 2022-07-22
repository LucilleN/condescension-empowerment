# from talkdown_talkup_matching import get_talkup_matched_samples
from utils import (
    read_talkdown, 
    read_filtered_reddit, 
    read_filtered_reddit_with_metadata,
    read_VAD_scores, 
    read_concreteness, 
    get_sentence_lexicon_score, 
    read_LIWC_lexicon, 
    get_LIWC_count,
    load_or_generate_dataframe,
    save_descriptive_stats,
    print_model_summaries,
    remove_outliers,
    plot_data,
    get_talkup_matched_samples,
    get_talkup_highest_power,
    get_talkup_random,
    get_talkup_highest_scores)
import statsmodels.formula.api as smf
import statsmodels.stats.weightstats as stattests
import pandas as pd


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

if __name__ == "__main__":

    ### Read TalkDown data as condescending set
    talkdown = read_talkdown()
    print(f"len(talkdown): {len(talkdown)}")
    ### Read filtered Reddit scrape as empowering set
    talkup_full = read_filtered_reddit()
    print(f"len(talkup_full): {len(talkup_full)}")

    talkup_highest_power = get_talkup_highest_power(talkup_full, len(talkdown))
    print(f"len(talkup_highest_power): {len(talkup_highest_power)}")

    talkup_matched = get_talkup_matched_samples(talkdown, talkup_full)
    print(f"len(talkup_matched): {len(talkup_matched)}")

    talkup_random = get_talkup_random(talkup_full, len(talkdown))
    print(f"len(talkup_random): {len(talkup_random)}")

    talkup_full_with_metadata = read_filtered_reddit_with_metadata()
    talkup_highest_scores = get_talkup_highest_scores(talkup_full_with_metadata, len(talkdown))

    ### Load VAD lexicon
    sentiment_scores = read_VAD_scores("v") # v for valence
    agency_scores = read_VAD_scores("a") # a for agency
    power_scores = read_VAD_scores("d") # d for dominance

    ### Load concreteness lexicon
    concreteness_scores = read_concreteness()

    ### Load LIWC
    liwc_words_by_category = read_LIWC_lexicon()

    # ### Feature extraction
    # X = [] # a list of lists, where rows are samples and columns are features
    # y = [] # a list of 0's and 1's corresponding to the label of each sample. 0 = condescension, 1 = empowerment

    data = load_or_generate_dataframe(talkdown, talkup_full, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category)
    data_abridged = load_or_generate_dataframe(talkdown, talkup_highest_power, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category, abridged=True)

    # save_descriptive_stats(data, 'descriptive_stats/unabridged.csv')
    # save_descriptive_stats(data_abridged, 'descriptive_stats/abridged.csv')

    # data_no_outliers = remove_outliers(data)
    # data_abridged_no_outliers = remove_outliers(data_abridged)
    # save_descriptive_stats(data_no_outliers, 'descriptive_stats/unabridged_no_outliers.csv')
    # save_descriptive_stats(data_abridged_no_outliers, 'descriptive_stats/abridged_no_outliers.csv')

    # different_datasets = {
    #     'Unabridged Data': data,
    #     'Abridged Data': data_abridged,
    #     'Unabridged Data w/o Outliers': data_no_outliers,
    #     'Abridged Data w/o Outliers': data_abridged_no_outliers
    # }
    # 
    # for dataset_name, dataset in different_datasets.items():
    #     # Plot a few selected columns
    #     plot_data(
    #         plot_type="boxplot",
    #         data=dataset, 
    #         fig_title=dataset_name, 
    #         subplot_names=['power', 'agency', 'sentiment', 'concreteness', "anger_count", "social_count", "relig_count", "sexual_count","humans_count"], 
    #         num_rows=3,
    #         num_cols=3
    #     )
    #     plot_data(
    #         plot_type="pdf",
    #         data=dataset, 
    #         fig_title=dataset_name, 
    #         subplot_names=['power', 'agency', 'sentiment', 'concreteness', "anger_count", "social_count", "relig_count", "sexual_count","humans_count"], 
    #         num_rows=3,
    #         num_cols=3
    #     )

    models = {}

    # # Trying four different datasets with simplest model
    # lr_model_1 = smf.logit("is_empowering ~ power + agency + sentiment + concreteness", data=data).fit()
    # models['VAD, CONCRETENESS, NO INTERACTIONS, FULL DATA'] = lr_model_1
    # lr_model_2 = smf.logit("is_empowering ~ power + agency + sentiment + concreteness", data=data_abridged).fit()
    # models['VAD, CONCRETENESS, NO INTERACTIONS, ABRIDGED DATA'] = lr_model_2
    # lr_model_3 = smf.logit("is_empowering ~ power + agency + sentiment + concreteness", data=data_abridged_no_outliers).fit()
    # models['VAD, CONCRETENESS, NO INTERACTIONS, ABRIDGED DATA NO OUTLIERS'] = lr_model_3
    
    # # Adding interactions
    # lr_model_4 = smf.logit("is_empowering ~ power * agency * sentiment * concreteness", data=data).fit()
    # models['VAD, CONCRETENESS, WITH INTERACTIONS, FULL DATA'] = lr_model_4

    # # Adding LIWC features
    # lr_model_5 = smf.logit("is_empowering ~ power + agency + sentiment + concreteness + anger_count + social_count + relig_count + sexual_count + humans_count", data=data).fit()
    # models['VAD, CONCRETENESS, AND LIWC COUNTS, NO INTERACTIONS, FULL DATA'] = lr_model_5

    # Best performing model so far -- now we want to try different ways of trimming the empowering set to 2600 examples
    data_highest_power = load_or_generate_dataframe(talkdown, talkup_highest_power, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category, abridged=True)
    lr_model_6 = smf.logit("is_empowering ~ power + agency + sentiment + concreteness + anger_count + social_count + relig_count + sexual_count + humans_count", data=data_abridged).fit()
    models['VAD, CONCRETENESS, AND LIWC COUNTS, NO INTERACTIONS, HIGHEST POWER DATA'] = lr_model_6

    # Trying with embedding-matched data
    data_matched = load_or_generate_dataframe(talkdown, talkup_matched, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category, matched=True)
    lr_model_7 = smf.logit("is_empowering ~ power + agency + sentiment + concreteness + anger_count + social_count + relig_count + sexual_count + humans_count", data=data_matched).fit()
    models['VAD, CONCRETENESS, AND LIWC COUNTS, NO INTERACTIONS, EMBEDDING-MATCHED DATA'] = lr_model_7

    # Trying with randomly sampled data
    data_random = load_or_generate_dataframe(talkdown, talkup_random, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category)
    lr_model_8 = smf.logit("is_empowering ~ power + agency + sentiment + concreteness + anger_count + social_count + relig_count + sexual_count + humans_count", data=data_random).fit()
    models['VAD, CONCRETENESS, AND LIWC COUNTS, NO INTERACTIONS, RANDOM DATA'] = lr_model_8

    # Trying with posts with highest score
    data_highest_score = load_or_generate_dataframe(talkdown, talkup_highest_scores, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category)
    lr_model_9 = smf.logit("is_empowering ~ power + agency + sentiment + concreteness + anger_count + social_count + relig_count + sexual_count + humans_count", data=data_highest_score).fit()
    models['VAD, CONCRETENESS, AND LIWC COUNTS, NO INTERACTIONS, HIGHEST SCORE DATA'] = lr_model_9

    # Trying with post with most awards
    
    print_model_summaries(models)