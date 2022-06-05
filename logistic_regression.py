from utils import (
    read_talkdown, 
    read_filtered_reddit, 
    read_VAD_scores, 
    read_concreteness, 
    get_sentence_lexicon_score, 
    read_LIWC_lexicon, 
    get_LIWC_count)
import statsmodels.formula.api as smf
import pandas as pd
from os.path import exists


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

def load_or_generate_data(condescending_set, empowering_set, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category):
    data = [] 
    if exists("data.pkl"):
        print("loading data...")
        data = pd.read_pickle("data.pkl")
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
        data.to_pickle("data.pkl")
    return data

if __name__ == "__main__":

    ### Read TalkDown data as condescending set
    condescending_set = read_talkdown()
    ### Read filtered Reddit scrape as empowering set
    empowering_set = read_filtered_reddit()

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

    lr_model_1 = smf.logit("is_empowering ~ power + agency + sentiment + concreteness", data=data).fit()
    
    lr_model_2 = smf.logit("is_empowering ~ power * agency * sentiment * concreteness", data=data).fit()
    
    lr_model_3 = smf.logit("is_empowering ~ power + agency + sentiment + concreteness + anger_count + social_count + relig_count + sexual_count + humans_count", data=data).fit()
   
    # This one fails to converge
    # lr_model_4 = smf.logit("is_empowering ~ power * agency * sentiment * concreteness * anger_count * social_count * relig_count * sexual_count * humans_count", data=data).fit()
    # lr_model_4 = smf.logit("is_empowering ~ power * agency * sentiment * concreteness * anger_binary * social_binary * relig_binary * sexual_binary * humans_binary", data=data).fit()
    
    # Trying to exclude anger because it was not a significant predictor, as well as social_count and sexual_count because they were significant but had small beta's
    # Still doesn't converge for all of the below
    # lr_model_4 = smf.logit("is_empowering ~ power * agency * sentiment * concreteness * relig_count * humans_count", data=data).fit()
    # lr_model_4 = smf.logit("is_empowering ~ power * agency * sentiment * concreteness * relig_count * humans_count", data=data).fit(maxiter=100)
    # lr_model_4 = smf.logit("is_empowering ~ power * agency * sentiment * concreteness * humans_count", data=data).fit(maxiter=100)

    print("\n\n######### MODEL 1: VAD, CONCRETENESS, NO INTERACTIONS #########\n")
    print(lr_model_1.summary())

    print("\n\n######### MODEL 2: VAD, CONCRETENESS, WITH INTERACTIONS #########\n")
    print(lr_model_2.summary())

    print("\n\n######### MODEL 3: VAD, CONCRETENESS, AND LIWC COUNTS, NO INTERACTIONS #########\n")
    print(lr_model_3.summary())

    # This one fails to converge
    # print("\n\n######### MODEL 4: VAD, CONCRETENESS, AND LIWC COUNTS, WITH INTERACTIONS #########\n")
    # print(lr_model_4.summary())