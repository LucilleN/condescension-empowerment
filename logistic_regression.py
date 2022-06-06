from utils import (
    read_talkdown, 
    read_filtered_reddit, 
    read_VAD_scores, 
    read_concreteness, 
    get_sentence_lexicon_score, 
    read_LIWC_lexicon, 
    get_LIWC_count,
    load_or_generate_dataframe,
    save_descriptive_stats,
    plot_data_boxplots,
    print_model_summaries,
    remove_outliers)
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

    models = {}

    lr_model_1 = smf.logit("is_empowering ~ power + agency + sentiment + concreteness", data=data).fit()
    models['VAD, CONCRETENESS, NO INTERACTIONS, FULL DATA'] = lr_model_1
    lr_model_2 = smf.logit("is_empowering ~ power + agency + sentiment + concreteness", data=data_abridged_no_outliers).fit()
    models['VAD, CONCRETENESS, NO INTERACTIONS, ABRIDGED DATA NO OUTLIERS'] = lr_model_2
    lr_model_3 = smf.logit("is_empowering ~ power * agency * sentiment * concreteness", data=data).fit()
    models['VAD, CONCRETENESS, WITH INTERACTIONS, FULL DATA'] = lr_model_3
    lr_model_4 = smf.logit("is_empowering ~ power + agency + sentiment + concreteness + anger_count + social_count + relig_count + sexual_count + humans_count", data=data).fit()
    models['VAD, CONCRETENESS, AND LIWC COUNTS, NO INTERACTIONS, FULL DATA'] = lr_model_4

    print_model_summaries(models)