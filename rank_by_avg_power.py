import json
import pandas as pd
from utils import read_talkdown

### Get power scores
power_scores = {}
with open('lexicons/NRC-VAD-Lexicon-Aug2018Release/OneFilePerDimension/d-scores.txt') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        word, score = line.split("\t")
        power_scores[word] = float(score)
        # print(f"power_scores[{word}] = {float(score)}")

def get_sentences_with_power_scores(sentences):
    sentences_with_power = []
    for sentence in sentences:
        # print(f"SENTENCE: {sentence}")
        word_power_scores = []
        for word in sentence.split():
            if word in power_scores:
                # print("found a word in the power scores")
                word_power_scores.append(power_scores[word])
        # should I just skip anything that doesn't have any token in the power lexicon?
        if len(word_power_scores) == 0: 
            continue
        sentence_avg_power = sum(word_power_scores) / len(word_power_scores) if len(word_power_scores) > 0 else None
        # print(f"{sentence} :: {sentence_avg_power}")
        sentences_with_power.append({'sentence': sentence, 'power': sentence_avg_power})

    # print(len(sentences_with_power))
    return sentences_with_power
    

### Read clean data
clean_set = pd.read_pickle('data/veiled-toxicity-detection/resources/processed_dataset/clean_train.pkl')
# This is a list of tuples where the second element is always None, so 
# just flattening it to a list of the strings
clean_set = [sentence for sentence, _ in clean_set]
clean_sentences_with_power = get_sentences_with_power_scores(clean_set)
clean_sorted_by_power = sorted(clean_sentences_with_power, key=lambda x: x['power'], reverse=True) 
# for x in clean_sorted_by_power:
#     print(x)


### Read condescending data from TalkDown
condescending_set = read_talkdown()

condescending_sentences_with_power = get_sentences_with_power_scores(condescending_set)
condescending_sorted_by_power = sorted(condescending_sentences_with_power, key=lambda x: x['power'], reverse=True) 
# for x in condescending_sorted_by_power:
#     print(x)


### Trim the longer dataset and find the average of power scores across all of them
clean_sorted_by_power_trimmed = clean_sorted_by_power[:len(condescending_sorted_by_power)]
for x in clean_sorted_by_power_trimmed:
    print(x)
# for x in clean_sorted_by_power:
#     print(x)

print(f"number of clean (trimmed): {len(clean_sorted_by_power_trimmed)}")
print(f"number of condescending: {len(condescending_sorted_by_power)}")

clean_avg_power_total = sum([x['power'] for x in clean_sorted_by_power]) / len(clean_sorted_by_power)
clean_avg_power_trimmed = sum([x['power'] for x in clean_sorted_by_power_trimmed]) / len(clean_sorted_by_power_trimmed)
print(f"clean average power (total): {clean_avg_power_total}")
print(f"clean average power (trimmed): {clean_avg_power_trimmed}")
clean_max_total = clean_sorted_by_power[0]['power']
clean_min_total = clean_sorted_by_power[-1]['power']
print(f'clean range (total): [{clean_max_total}, {clean_min_total}]')
clean_max_trimmed = clean_sorted_by_power_trimmed[0]['power']
clean_min_trimmed = clean_sorted_by_power_trimmed[-1]['power']
print(f'clean range (trimmed): [{clean_max_trimmed}, {clean_min_trimmed}]')

condescending_avg_power = sum([x['power'] for x in condescending_sorted_by_power]) / len(condescending_sorted_by_power)
print(f"condescending average power: {condescending_avg_power}")
print(f"condescending range: [{condescending_sorted_by_power[0]['power']}, {condescending_sorted_by_power[-1]['power']}]")


