from utils import read_talkdown, read_filtered_reddit, read_VAD_scores

### Read TalkDown data as condescending set
condescending_set = read_talkdown()
### Read filtered Reddit scrape as empowering set
empowering_set = read_filtered_reddit()


### Load VAD lexicon
sentiment_scores = read_VAD_scores("v") # v for valence
agency_scores = read_VAD_scores("a") # a for agency
power_scores = read_VAD_scores("d") # d for dominance


### Load concreteness lexicon
# load LIWC


# feature extraction
