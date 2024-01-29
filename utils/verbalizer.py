VERBALIZER = {
    "sst2": ['positive', 'negative'],
    "intents": ["no", "yes"],
    # "sst5": ["very positive", "positive", "neutral", "negative", "very negative"]
    "sst5": ["terrible", "bad", "neutral", "good", "great"],
    "rte": ['yes', "no"],
    "anli": ['yes', 'maybe', 'no'],
    "cause_and_effect": ['yes', "no"],
    "causal_judgment": ['yes', "no"],
    "manifestos": ['other', "external", "democracy", "political", "economy", "welfare", "fabric", 'group'],
    'hate_speech': ['support', 'neutral', 'hate']
    # "manifestos": ['other', "external relations", "freedom and democracy", "political system", "economy", "welfare and quality of life", "fabric of society", 'social groups']
}

SST5_NEW_LABEL_MAP = {
    "very positive": "great",
    "positive": "good",
    "neutral": "neutral",
    "negative": "bad",
    "very negative": "terrible"
}

MANIFESTOS_MAPPING = {
    "Other": "Other",
    "External Relations": "External",
    "Freedom and Democracy": "Democracy",
    "Political System": "Political",
    "Economy": "Economy",
    "Welfare and Quality of Life": "Welfare",
    "Fabric of Society": "Fabric",
    "Social Groups": "Group"
}

RTE_ID2VERB = {VERBALIZER['rte'].index(i): i for i in VERBALIZER['rte']}
ANLI_ID2VERB = {VERBALIZER['anli'].index(i): i for i in VERBALIZER['anli']}