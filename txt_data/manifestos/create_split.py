import os
import json

MANIFESTOS_MAPPING = {
    0: "Other",
    1: "External Relations",
    2: "Freedom and Democracy",
    3: "Political System",
    4: "Economy",
    5: "Welfare and Quality of Life",
    6: "Fabric of Society",
    7: "Social Groups"
}

label_space = list(MANIFESTOS_MAPPING.values())

file_lists = os.listdir('./')

all_data = []
for file in file_lists:
    if file.endswith('.txt'):
        with open(file, 'r') as f:
            data = f.readlines()
        all_data += data

sent_list = []
label_list = []
sent_dict = {k: [] for k in label_space}
repeat = []
repeat_label = []
for item in all_data:
    sent, _, label = item.split('\t')
    label = label.strip("\"\n")
    sent = sent.strip("\"")
    if sent not in sent_list:
        if label == "NA":
            continue
        else:
            sent_list.append(sent)
            label_list.append(MANIFESTOS_MAPPING[int(label)])
            sent_dict[MANIFESTOS_MAPPING[int(label)]].append(sent)
    else:
        idx = sent_list.index(sent)
        if MANIFESTOS_MAPPING[int(label)] == label_list[idx]:
            continue
        _ = sent_list.pop(idx)
        _ = label_list.pop(idx)
        repeat.append(sent)
        repeat_label.append(label.strip())

sent_num = len(set(sent_list))

balance_train_500 = {k: v[:100] for k, v in sent_dict.items()}
balance_test = {k: v[-100:] for k, v in sent_dict.items()}
imbalanced_train_500 = {}
imbalanced_test = {}
for k, v in sent_dict.items():
    if k in ['Fabric of Society', 'Economy']:
        imbalanced_train_500[k] = v[:200]
        imbalanced_test[k] = v[-200:]
    else:
        imbalanced_train_500[k] = v[:70]
        imbalanced_test[k] = v[-70:]

data = []
for k, v in imbalanced_train_500.items():
    for v_item in v:
        item = {
            "sentence": v_item,
            "label": k
        }
        data.append(item)

with open("train_imbalance_500.json", 'w') as f:
    json.dump(data, f, indent=4)
print(1)