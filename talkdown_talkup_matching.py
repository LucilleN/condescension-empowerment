from os.path import exists
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
    print_model_summaries,
    remove_outliers,
    plot_data)
import statsmodels.formula.api as smf
import statsmodels.stats.weightstats as stattests
import pandas as pd
from sentence_transformers import SentenceTransformer    
from numpy import dot
from numpy.linalg import norm

# using this library: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
# errors with installing the sentence_transformers library can be solved with this solution: https://github.com/UKPLab/sentence-transformers/issues/128
# model = SentenceTransformer('all-mpnet-base-v2')



def get_sentence_embeddings(sentences):
    # using this library for the model: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
    # errors with installing the sentence_transformers library can be solved with this solution: https://github.com/UKPLab/sentence-transformers/issues/128
    model = SentenceTransformer('all-mpnet-base-v2')
    sentence_embeddings = []
    for sentence in sentences:
        print("Calling model.encode on one sentence")
        embedding = model.encode(sentence)
        sentence_embeddings.append((sentence, embedding))
    return sentence_embeddings

def cosine_similarity(vector1, vector2):
    return dot(vector1, vector2) / (norm(vector1) * norm(vector2))

def get_most_similar_sentence(s1, embed1, sentence_embeddings):
    highest_similarity = 0
    matched_sentence = None
    for index, row in sentence_embeddings.iterrows():
        s2 = row['sentence']
        embed2 = row['embedding']
        similarity = cosine_similarity(embed1, embed2)
        # print(f"found cosine similarity of {similarity}")
        if similarity > highest_similarity:
            highest_similarity = similarity
            print(f"Highest similarity is now {highest_similarity}")
            matched_sentence = s2
    
    # print(f"Sentence 1: {s1}")
    # print(f"MATCHED WITH")
    # print(f"Sentence 2: {matched_sentence}")
    return matched_sentence

def get_talkup_matched_samples(condescending_set, empowering_set):
    talkdown_embeds = None
    talkup_embeds = None

    if exists("talkdown_embeddings.pkl"):
        print("loading data...")
        talkdown_embeds = pd.read_pickle("talkdown_embeddings.pkl")
    else:
        print("Calling get_sentence_embeddings for talkdown")
        talkdown_embeds = get_sentence_embeddings(condescending_set)
        # print("TALKDOWN EMBEDS")
        # for item in talkdown_embeds:
        #     print("\n\n")
        #     print(item)
        talkdown_embeds = pd.DataFrame(talkdown_embeds, columns =['sentence', 'embedding'])
        talkdown_embeds.to_pickle("talkdown_embeddings.pkl")
    
    if exists("talkup_embeddings.pkl"):
        print("loading data...")
        talkup_embeds = pd.read_pickle("talkup_embeddings.pkl")
    else:
        print("Calling get_sentence_embeddings for talkup")
        talkup_embeds = get_sentence_embeddings(empowering_set)
        talkup_embeds = pd.DataFrame(talkup_embeds, columns =['sentence', 'embedding'])
        talkup_embeds.to_pickle("talkup_embeddings.pkl")

    talkup_matched = []
   
    if exists("talkup_matched.csv"):
        df = pd.read_csv("talkup_matched.csv")
        talkup_matched = df.values.reshape(-1).tolist()
    else:
        for index, row in talkdown_embeds.iterrows():
            # print(tup)
            s1 = row['sentence']
            embed1 = row['embedding']
            print(f"TalkDown sentence: {s1}") #, \nembedding: {embed1}")
            matched_sentence = get_most_similar_sentence(s1, embed1, talkup_embeds)
            # print(f"\nTalkDown sentence: {s1}")
            print(f"Matched with TalkUp sentence: {matched_sentence}")
            talkup_matched.append(matched_sentence)

        pd.DataFrame(talkup_matched).to_csv("talkup_matched.csv")
    
    return talkup_matched

if __name__ == "__main__":

    ### Read TalkDown data as condescending set
    condescending_set = read_talkdown()
    print(f"len(condescending_set): {len(condescending_set)}")
    ### Read filtered Reddit scrape as empowering set
    empowering_set = read_filtered_reddit()

    talkup_matched = get_talkup_matched_samples(condescending_set, empowering_set)

    empowering_set_abridged = read_filtered_reddit(abridged=True, k=len(condescending_set))
    # print(f"len(empowering_set): {len(empowering_set)}")
    print(f"len(empowering_set_abridged): {len(empowering_set_abridged)}")

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

    # data = load_or_generate_dataframe(condescending_set, empowering_set, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category)
    data_abridged = load_or_generate_dataframe(condescending_set, empowering_set_abridged, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category, abridged=True)
    data_matched = load_or_generate_dataframe(condescending_set, talkup_matched, power_scores, agency_scores, sentiment_scores, concreteness_scores, liwc_words_by_category, matched=True)
    print("POTATO POTATO POTATO")
    print(f"len(talkup_matched): {len(talkup_matched)}")
    print(f"len(data_matched): {len(data_matched)}")


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
    lr_model_6 = smf.logit("is_empowering ~ power + agency + sentiment + concreteness + anger_count + social_count + relig_count + sexual_count + humans_count", data=data_abridged).fit()
    models['VAD, CONCRETENESS, AND LIWC COUNTS, NO INTERACTIONS, ABRIDGED DATA'] = lr_model_6
    
    lr_model_7 = smf.logit("is_empowering ~ power + agency + sentiment + concreteness + anger_count + social_count + relig_count + sexual_count + humans_count", data=data_matched).fit()
    models['VAD, CONCRETENESS, AND LIWC COUNTS, NO INTERACTIONS, MATCHED DATA'] = lr_model_7
    

    print_model_summaries(models)


    print(f"len(talkup_matched): {len(talkup_matched)}")
    print(f"len(data_matched): {len(data_matched)}")

    print(f"len(empowering_set_abridged): {len(empowering_set_abridged)}")
    print(f"len(data_abridged): {len(data_abridged)}")

    # print(talkup_matched)
    # print("\n\n\n")
    # print(empowering_set_abridged)