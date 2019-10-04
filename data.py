import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import random

class AceData(Dataset):
    """"""
    def __init__(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            examples = [json.loads(ex) for ex in f.read().splitlines()]

        self.examples = prepare_examples(examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # ex = json.loads(ex)
        # doc_id = ex["doc_id"]
        # sentences = ex["words"]
        # events = ex["events"]
        # labels = ex["labels"]

        # sentence, events, doc_id
        return ex[0], ex[1], ex[2]

class ConllData(Dataset):
    """"""
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex[0], ex[1] # sentences, labels

# def read_conll(filename):
#     with open(filename, "r", encoding="utf-8") as f:
#         lines = f.read().splitlines()

#     examples = []
#     words = []
#     sent = []
#     labels = []
#     label_to_id = dict()
#     j = 0
#     for (i, line) in enumerate(lines):
#         line = line.strip().split("\t") # Split word and label
#         if line[0] == "SAMPLE_START":
#             continue

#         elif line[0] == "" or line[0] == "[SEP]" : # End of sample -> Each sample is a sentence
#             examples.append((words, labels))
#             j += 1
#             words = []
#             sent = []
#             labels = []
#             continue

#         # elif line[0] in ["\x91", "\x92", "\x97"]:
#         #     continue
#         # elif line[0] == "[SEP]":
#         #     words.append(sent)
#         #     sent = []

#         else:
#             sent.append(line[0])
#             labels.append(line[1])
#             if line[1] not in label_to_id.keys():
#                 label_to_id[line[1]] = len(label_to_id)

#     return examples, label_to_id

def read_conll(filename, doc_level=False):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    examples = []
    all_words = []
    all_labels = []
    sent = []
    sent_labels = []
    label_to_id = dict()
    j = 0
    for (i, line) in enumerate(lines):
        line = line.strip().split("\t") # Split word and label
        if line[0] == "SAMPLE_START":
            continue

        elif line[0] == "": # End of document, also end of last sentence of document
            if doc_level:
                examples.append((all_words, all_labels))
            else:
                examples.append((sent, sent_labels))

            j += 1
            all_words = []
            sent = []
            sent_labels = []
            all_labels = []
            continue

        # elif line[0] in ["\x91", "\x92", "\x97"]:
        #     continue
        elif line[0] == "[SEP]": # End of sentence
            if doc_level:
                all_words.extend(sent)
                all_labels.extend(sent_labels)
            else:
                examples.append((sent, sent_labels))

            sent = []
            sent_labels = []

        else:
            sent.append(line[0])
            if line[1] not in label_to_id.keys():
                label_to_id[line[1]] = len(label_to_id) + 1 # start from 1, because we use 0 as pad id

            sent_labels.append(label_to_id[line[1]])

    return examples, label_to_id

def pad_labels(labels):
    max_len = max([len(label) for label in labels])
    for i in range(len(labels)):
        labels[i] = labels[i] + [0]*max(max_len - len(labels[i]), 0) # 0 is pad id

    return labels

def conll_collate(batch, doc_level=False):
    # If not doc_level -> each item in sentences is a list and is independent of each other.
    # If doc_level -> each item in sentences is a list of lists representing one document.
    # Can see this in code in read_conll

    sentences = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    if doc_level: # We don't care about seperating sentences in train time, so we just open them up
        for i in range(len(sentences)):
            labels[i] = [label for label_list in labels[i] for label in label_list]
            sentences[i] = [sentence for sentence_list in sentences[i] for sentence in sentence_list]

    # print(labels)
    labels = pad_labels(labels)
    # print(labels)
    labels = torch.tensor(labels, dtype=torch.long)

    return sentences, labels

def get_dict(d, key):
    if key in d.keys():
        return d[key]
    else:
        return []

def prepare_examples(examples):
    all_examples = []
    for ex in examples:
        doc_id = ex["doc_id"]
        sentences = ex["words"]
        events = ex["events"]
        # labels = ex["labels"]

        # sent_idx => [trigger_list, argument_list]
        events_data = {}
        for i,event in enumerate(events):
            # event["sent_idx"] are zero-ordered
            # start-end indexes are document beggining oriented instead of sentence beginning. -> Need to normalize. -> TODO: fix these in json files we read

            # Sentence index, (Start-end index, event_type:event_subtype)
            # sentence index needed for ordering later
            trigger = ((event["trigger"]["start"], event["trigger"]["end"]),
                       event["event_type"] + ":" + event["event_subtype"])

            # TODO: if we compare event_label's of arguments, we need to add them to predicted arguments somehow
            # TODO: maybe for predicted, assign the first entity to it
            arguments = []
            for arg in event["arguments"]:
                # list of ((start, end), trigger_label, role)
                # only need trigger label to compare -> referred in function invert_arguments in https://github.com/dwadden/dygiepp/blob/master/dygie/training/event_metrics.py
                arguments.append(((arg["start"], arg["end"]), trigger[1], arg["role"]))

            if event["sent_idx"] in events_data.keys():
                new_events = [events_data[event["sent_idx"]][0] + [trigger],
                              events_data[event["sent_idx"]][1] + arguments]
                events_data[event["sent_idx"]] = new_events
            else:
                events_data[event["sent_idx"]] = [[trigger], arguments]

        for j, sent in enumerate(sentences):
            all_examples.append((sent, get_dict(events_data, j), doc_id)) # maybe add sent_index as well

    return all_examples

def ace_collate(batch):
    sentences = [item[0] for item in batch]
    events = [item[1] for item in batch]
    doc_ids = [item[2] for item in batch]
    # # TODO: Consider what to do with these
    # labels = [item[3] for item in batch] # doc labels

    return sentences, events, doc_ids

def get_dataloaders(args):
    examples, label_to_id = read_conll(args.train_filename, doc_level=args.doc_level)
    random.seed(args.seed)
    random.shuffle(examples)

    train = DataLoader(dataset=ConllData(examples), batch_size=args.train_batch_size,
                       shuffle=True, collate_fn=lambda x: conll_collate(x, doc_level=args.doc_level))
    dev = DataLoader(dataset=AceData(args.dev_filename), batch_size=args.eval_batch_size,
                     collate_fn=ace_collate)
    test = DataLoader(dataset=AceData(args.test_filename), batch_size=args.eval_batch_size,
                      collate_fn=ace_collate)

    return train, dev, test, label_to_id
    # return train, label_to_id
