import json


def read_talkdown():
    """
    Reads condescending data from TalkDown.
    Returns:
        A list of strings, where each string is the 'quotedpost' field 
    """
    condescending_set = []
    with open("data/talkdown/data/balanced_train.jsonl") as f:
        json_list = list(f)
    for json_str in json_list:
        data = json.loads(json_str)
        if data['label']: # if it's labeled as True for condescending
            condescending_set.append(data['quotedpost']) # This is just the part that someone replied to pointing out that it's condescending
            # condescending_set.append(data['post']) # This is the entire post, which contains quotedpost plus more context
    return condescending_set