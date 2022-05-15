import pandas as pd

clean_set = pd.read_pickle(r'veiled-toxicity-detection/resources/processed_dataset/clean_train.pkl')
for sentence in clean_set:
    print(sentence) 