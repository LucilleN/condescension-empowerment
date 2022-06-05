from utils import read_talkdown, read_filtered_reddit, read_VAD_scores, read_concreteness, get_sentence_lexicon_score, read_LIWC_lexicon, get_LIWC_count
import statsmodels.formula.api as smf
import pandas as pd


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

def get_feature_vector(sentence):
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
    
    print(feature_vector)
    return feature_vector

data = []

for sentence in condescending_set:
    data_point = [0] + get_feature_vector(sentence)
    data.append(data_point)

for sentence in empowering_set:
    data_point = [1] + get_feature_vector(sentence)
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

lr_model_1 = smf.logit("is_empowering ~ power + agency + sentiment + concreteness", data=data).fit()
lr_model_2 = smf.logit("is_empowering ~ power * agency * sentiment * concreteness", data=data).fit()

print("\n\n######### MODEL 1: VAD, CONCRETENESS, NO INTERACTIONS #########\n")
print(lr_model_1.summary())

print("\n\n######### MODEL 2: VAD, CONCRETENESS, WITH INTERACTIONS #########\n")
print(lr_model_2.summary())