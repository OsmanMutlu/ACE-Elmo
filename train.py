import torch
from allennlp.modules.elmo import batch_to_ids, Elmo
from data import get_dataloaders
from model import SequenceTagger
import numpy as np
import ipdb
from tqdm import tqdm, trange
from event_metrics import score_triggers, score_arguments

from conlleval import evaluate2

class Config():
    def __init__(self):
        self.hidden_size_lstm = 1024
        self.ntags = 0
        self.dropout = 0.1
        self.dim_elmo = 1024
        # self.train_filename = "/scratch/users/omutlu/ace2005/Token/Life/only_trigger_train.txt"
        # # self.dev_filename = "/scratch/users/omutlu/ace2005/Token/dev_with_words.json"
        # # self.test_filename = "/scratch/users/omutlu/ace2005/Token/test_with_words.json"
        # self.dev_filename = "/scratch/users/omutlu/ace2005/Token/Life/only_trigger_dev.txt"
        # self.test_filename = "/scratch/users/omutlu/ace2005/Token/Life/only_trigger_test.txt"
        self.train_filename = "/scratch/users/omutlu/20190325-CLEF-ProtestNews-Train-Dev-output/Token/train.txt"
        self.dev_filename = "/scratch/users/omutlu/20190325-CLEF-ProtestNews-Train-Dev-output/Token/dev.txt"
        self.test_filename = "/scratch/users/omutlu/20190325-CLEF-ProtestNews-Train-Dev-output/Token/test.txt"
        self.train_batch_size = 32
        self.eval_batch_size = 8
        self.doc_level = False
        self.use_gpu = True
        self.seed = 42
        self.num_epochs = 10
        self.options_file = "/scratch/users/omutlu/Pytorch_ELMO/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        self.weight_file = "/scratch/users/omutlu/Pytorch_ELMO/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.do_train = False
        self.do_test = True
        self.output_file = "/scratch/users/omutlu/Pytorch_ELMO/trigger_model.pt"


def mask_targets(targets, sequence_lengths, batch_first=False):
    """ Masks the targets """
    if not batch_first:
         targets = targets.transpose(0,1)
    t = []
    for l, p in zip(targets,sequence_lengths):
        t.append(l[:p].data.tolist())
    return t

# TODO: check if this function is valid
# TODO: Also check if pad in label dict is causing some stuff to go wrong
def evaluate(sample_num, el, id_to_label, label_prefix):
    pred_triggers = []
    pred_arguments = []
    prev_label = "O"
    trigger_label = ""

    # trigger selection algorithm for arguments.
    # TODO: Think of a more sophisticated way of doing this
    for label in el:
        lbl = id_to_label[label]
        if lbl != "O":
            if lbl[2:] in trigger_list:
                trigger_label = lbl[2:]
                break

    for idx,label in enumerate(el):
        label = id_to_label[label]
        if "B-" in label:
            if prev_label != "O":
                if prev_label[2:] in trigger_list:
                    pred_triggers.append(((start_idx, idx), label_prefix + prev_label[2:]))
                    trigger_label = prev_label[2:]
                else:
                    pred_arguments.append(((start_idx, idx), trigger_label, prev_label[2:]))

            start_idx = idx

        elif "I-" in label:
            if prev_label == "O":
                start_idx = idx
            elif label[2:] != prev_label[2:]: # label changed
                if prev_label[2:] in trigger_list:
                    pred_triggers.append(((start_idx, idx), label_prefix + prev_label[2:]))
                    trigger_label = prev_label[2:]
                else:
                    pred_arguments.append(((start_idx, idx), trigger_label, prev_label[2:]))

                start_idx = idx

        else:
            if prev_label != "O":
                if prev_label[2:] in trigger_list:
                    pred_triggers.append(((start_idx, idx), label_prefix + prev_label[2:]))
                    trigger_label = prev_label[2:]
                else:
                    pred_arguments.append(((start_idx, idx), trigger_label, prev_label[2:]))

        prev_label = label

    if len(events[sample_num]) != 0:
        # TODO: check if we only compare events that are in same sentence
        curr_trigger_results = score_triggers(pred_triggers, events[sample_num][0])
        curr_argument_results = score_arguments(pred_arguments, events[sample_num][1])
    else:
        curr_trigger_results = [0, len(pred_triggers), 0, 0]
        curr_argument_results = [0, len(pred_arguments), 0, 0]

    return curr_trigger_results, curr_argument_results


def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0

def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1

n_gpu = torch.cuda.device_count()
device = "cpu"

args = Config()

# *** LOAD DATA ***
train, dev, test, label_to_id = get_dataloaders(args)
# train, label_to_id = get_dataloaders(args)
args.ntags = len(label_to_id) + 1 # +1 for Pad at 0th index

if args.use_gpu and n_gpu > 0:
    device = "cuda"

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

model = SequenceTagger(args)
model.to(device)
optimizer = torch.optim.Adam(model.parameters())
if n_gpu > 1:
    model = torch.nn.DataParallel(model)

# trigger_list = ["Life:Be-Born", "Life:Die", "Life:Marry", "Life:Divorce", "Life:Injure"]
trigger_list = ["Be-Born", "Die", "Marry", "Divorce", "Injure"]

id_to_label = dict()
for i,label in enumerate(label_to_id.keys()):
    id_to_label[i+1] = label # i+1 because we start from 1 since pad is 0

elmo = Elmo(args.options_file, args.weight_file, 2, dropout=0)
elmo.to(device)

if args.do_train:
    best_f1 = 0.0
    for epoch in trange(args.num_epochs):
        train_loss = 0
        # total = 0
        # correct = 0
        for step, (sentences, label_ids) in enumerate(train):
            # if step < 910:
            #     continue
            # continue # TODO: remove after debug
            seq_lengths = [len(sent) for sent in sentences]
            char_ids = batch_to_ids(sentences)
            char_ids = char_ids.to(device)
            # Returns a dict with keys=["elmo_representations" -> [BxSx1024, BxSx1024], "mask" -> BxS]. in "elmo_representations" returns a list, but two tensors are copies of each other.
            embeddings = elmo(char_ids)
            word_input = embeddings['elmo_representations'][0].detach().requires_grad_(False)
            mask = embeddings['mask'].clone().detach()
            del embeddings
            word_input = word_input.to(device)
            mask = mask.to(device)
            label_ids = label_ids.to(device)
            loss, _ = model(word_input, mask=mask, labels=label_ids)

            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            model.zero_grad()

            # print(train_loss / ((step+1)*args.train_batch_size), end=", ")
            # print("%.4f -> %.4f"%(loss.item() / args.train_batch_size, train_loss / ((step+1)*args.train_batch_size)), end=", ")


        # TODO: you were gonna try evaluating by conll script!!! You changed dataloaders and left off there!
        all_trigger_results = [0, 0, 0, 0]
        all_argument_results = [0, 0, 0, 0]
        # all_gold_triggers, all_predicted_triggers, all_matched_trigger_ids, all_matched_trigger_classes = 0, 0, 0, 0
        # all_gold_arguments, all_predicted_arguments, all_matched_argument_ids, all_matched_argument_classes = 0, 0, 0, 0
        all_preds = []
        all_label_ids = np.array([])
        model.eval()
        # for step, (sentences, events, doc_ids) in enumerate(dev):
        for step, (sentences, label_ids) in enumerate(dev):
            seq_lengths = [len(sent) for sent in sentences]
            char_ids = batch_to_ids(sentences)
            char_ids = char_ids.to(device)
            embeddings = elmo(char_ids)
            word_input = embeddings['elmo_representations'][0].detach().requires_grad_(False)
            mask = embeddings['mask'].clone().detach()
            del embeddings
            word_input = word_input.to(device)
            mask = mask.to(device)
            preds = model(word_input, mask=mask)
            preds = [p[:i] for p, i in zip(preds, seq_lengths)] # in test time
            # preds = [p.item() for p in preds[:-1]] + preds[-1]

            all_preds.extend([id_to_label[p] if p != 0 else "O" for pred in preds for p in pred])
            all_label_ids = np.append(all_label_ids, label_ids[label_ids != 0])

        #     for sample_num,el in enumerate(preds):
        #         curr_trigger_results, curr_argument_results = evaluate(sample_num, el, id_to_label, "Life:")
        #         all_trigger_results = [x + y for x, y in zip(all_trigger_results, curr_trigger_results)]
        #         all_argument_results = [x + y for x, y in zip(all_argument_results, curr_argument_results)]

        # ti_precision, ti_recall, ti_f1 = compute_f1(all_trigger_results[1], all_trigger_results[0], all_trigger_results[2])
        # tc_precision, tc_recall, tc_f1 = compute_f1(all_trigger_results[1], all_trigger_results[0], all_trigger_results[3])

        # ai_precision, ai_recall, ai_f1 = compute_f1(all_argument_results[1], all_argument_results[0], all_argument_results[2])
        # ac_precision, ac_recall, ac_f1 = compute_f1(all_argument_results[1], all_argument_results[0], all_argument_results[3])

        conll_precision, conll_recall, conll_f1 = evaluate2([id_to_label[x] for x in all_label_ids.tolist()], all_preds)

        print("End of EPOCH %d" % epoch)
        # TODO: How do we decide to save?
        # curr_f1 = (ti_f1 + tc_f1 + ai_f1 + ac_f1) / 4
        curr_f1 = conll_f1
        if curr_f1 > best_f1:
            best_f1 = curr_f1
            print("Saving model...")
            model_to_save = model.module if hasattr(model, 'module') else model  # To handle multi gpu
            torch.save(model_to_save.state_dict(), args.output_file)

        print("Conll Results : precision -> %.4f, recall -> %.4f, f1 -> %.4f" %(conll_precision, conll_recall, conll_f1))
        # print("Trigger Identification : precision -> %.4f, recall -> %.4f, f1 -> %.4f" %(ti_precision, ti_recall, ti_f1))
        # print("Trigger Classification : precision -> %.4f, recall -> %.4f, f1 -> %.4f" %(tc_precision, tc_recall, tc_f1))
        # print("Argument Identification : precision -> %.4f, recall -> %.4f, f1 -> %.4f" %(ai_precision, ai_recall, ai_f1))
        # print("Argument Classification : precision -> %.4f, recall -> %.4f, f1 -> %.4f" %(ac_precision, ac_recall, ac_f1))

        model.train()

# TODO: Fix below traceback
# Traceback (most recent call last):
#   File "train.py", line 260, in <module>
#     curr_trigger_results, curr_argument_results = evaluate(sample_num, el, id_to_label, "Life:")
#   File "train.py", line 57, in evaluate
#     label = id_to_label[label]
# KeyError: 0

if args.do_test:
    model.load_state_dict(torch.load(args.output_file))
    model.to(device)
    model.eval()
    all_trigger_results = [0, 0, 0, 0]
    all_argument_results = [0, 0, 0, 0]
    # all_gold_triggers, all_predicted_triggers, all_matched_trigger_ids, all_matched_trigger_classes = 0, 0, 0, 0
    # all_gold_arguments, all_predicted_arguments, all_matched_argument_ids, all_matched_argument_classes = 0, 0, 0, 0
    # for step, (sentences, events, doc_ids) in enumerate(test):
    all_preds = []
    all_label_ids = np.array([])
    for step, (sentences, label_ids) in enumerate(test):
        seq_lengths = [len(sent) for sent in sentences]
        char_ids = batch_to_ids(sentences)
        char_ids = char_ids.to(device)
        embeddings = elmo(char_ids)
        word_input = embeddings['elmo_representations'][0].detach().requires_grad_(False)
        mask = embeddings['mask'].clone().detach()
        del embeddings
        word_input = word_input.to(device)
        mask = mask.to(device)
        preds = model(word_input, mask=mask)
        preds = [p[:i] for p, i in zip(preds, seq_lengths)] # in test time

        all_preds.extend([id_to_label[p] if p != 0 else "O" for pred in preds for p in pred])
        all_label_ids = np.append(all_label_ids, label_ids[label_ids != 0])

    conll_precision, conll_recall, conll_f1 = evaluate2([id_to_label[x] for x in all_label_ids.tolist()], all_preds)

    #     for sample_num,el in enumerate(preds):
    #         curr_trigger_results, curr_argument_results = evaluate(sample_num, el, id_to_label, "Life:")
    #         all_trigger_results = [x + y for x, y in zip(all_trigger_results, curr_trigger_results)]
    #         all_argument_results = [x + y for x, y in zip(all_argument_results, curr_argument_results)]

    print("Conll Results : precision -> %.4f, recall -> %.4f, f1 -> %.4f" %(conll_precision, conll_recall, conll_f1))
    # ti_precision, ti_recall, ti_f1 = compute_f1(all_trigger_results[1], all_trigger_results[0], all_trigger_results[2])
    # tc_precision, tc_recall, tc_f1 = compute_f1(all_trigger_results[1], all_trigger_results[0], all_trigger_results[3])

    # ai_precision, ai_recall, ai_f1 = compute_f1(all_argument_results[1], all_argument_results[0], all_argument_results[2])
    # ac_precision, ac_recall, ac_f1 = compute_f1(all_argument_results[1], all_argument_results[0], all_argument_results[3])

    # print("Trigger Identification : precision -> %.4f, recall -> %.4f, f1 -> %.4f" %(ti_precision, ti_recall, ti_f1))
    # print("Trigger Classification : precision -> %.4f, recall -> %.4f, f1 -> %.4f" %(tc_precision, tc_recall, tc_f1))
    # print("Argument Identification : precision -> %.4f, recall -> %.4f, f1 -> %.4f" %(ai_precision, ai_recall, ai_f1))
    # print("Argument Classification : precision -> %.4f, recall -> %.4f, f1 -> %.4f" %(ac_precision, ac_recall, ac_f1))

# TODO: check whole process from the beginning, including data generation
# TODO: train with all event types (Need to generalize trigger_list and its check!!!) -> our system
# TODO: generate a single BIO train file for general token extractor, and train on it -> general system
# TODO: Try out new paper's code at -> https://github.com/dwadden/dygiepp
