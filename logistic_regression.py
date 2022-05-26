from utils import read_talkdown, read_filtered_reddit, read_VAD_scores, read_concreteness, get_sentence_lexicon_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
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


### Feature extraction
X = [] # a list of lists, where rows are samples and columns are features
y = [] # a list of 0's and 1's corresponding to the label of each sample. 0 = condescension, 1 = empowerment

def get_feature_vector(sample):
    sentence_avg_power = get_sentence_lexicon_score(sample, power_scores)
    sentence_avg_agency = get_sentence_lexicon_score(sample, agency_scores)
    sentence_avg_sentiment = get_sentence_lexicon_score(sample, sentiment_scores)
    sentence_avg_concreteness = get_sentence_lexicon_score(sample, concreteness_scores)

    return [sentence_avg_power, sentence_avg_agency, sentence_avg_sentiment, sentence_avg_concreteness]


# for sentence in condescending_set: 
#     X.append(get_feature_vector(sentence))
#     y.append(0)

# for sentence in empowering_set:
#     X.append(get_feature_vector(sentence))
#     y.append(1)

data = []

for sentence in condescending_set:
    data_point = get_feature_vector(sentence)
    data_point.append(0)
    data.append(data_point)

for sentence in empowering_set:
    data_point = get_feature_vector(sentence)
    data_point.append(1)
    data.append(data_point)

# print(X)
# print(y)

# lr_model = LogisticRegression()

# lr_model.fit(X, y)

# predicted_labels = lr_model.predict(X)
# print(classification_report(y_true=y, y_pred=predicted_labels))

print(data)
print(len(data))
print(len(data[0]))
data = pd.DataFrame(data, columns = ['power', 'agency', 'sentiment', 'concreteness', 'is_empowering'])

# lr_model = smf.logit("is_empowering ~ power + agency + sentiment + concreteness", data=data).fit()
lr_model = smf.logit("is_empowering ~ power * agency * sentiment * concreteness", data=data).fit()

print(lr_model.summary())
