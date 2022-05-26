from utils import read_talkdown, read_filtered_reddit, read_VAD_scores, read_concreteness

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

### load LIWC


# feature extraction
for sentence in condescending_set: 
    sentence_avg_power = get_sentence_lexicon_score(sentence, power_scores)
    sentence_avg_agency = get_sentence_lexicon_score(sentence, agency_scores)
    sentence_avg_sentiment = get_sentence_lexicon_score(sentence, sentiment_scores)
    sentence_avg_concreteness = get_sentence_lexicon_score(sentence, concreteness_scores)
    # label 0 will mean condescension
    # label 1 will mean empowerment
