import torch
from allennlp.modules.elmo import batch_to_ids, Elmo
from data import get_dataloaders
from model import SequenceTagger
import numpy as np
import pdb
from tqdm import tqdm, trange

class Config():
    def __init__(self):
        self.hidden_size_lstm = 1024
        self.ntags = 0
        self.dropout = 0.1
        self.dim_elmo = 1024
        self.train_filename = "/scratch/users/omutlu/ace2005/Token/Life/only_trigger_train.txt"
        self.dev_filename = "/scratch/users/omutlu/ace2005/Token/dev_with_words.json"
        self.test_filename = "/scratch/users/omutlu/ace2005/Token/test_with_words.json"
        self.train_batch_size = 8
        self.eval_batch_size = 8
        self.doc_level = False
        self.use_gpu = False
        self.seed = 42
        self.num_epochs = 10
        self.options_file = "/scratch/users/omutlu/Pytorch_ELMO/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        self.weight_file = "/scratch/users/omutlu/Pytorch_ELMO/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.do_eval = True


def mask_targets(targets, sequence_lengths, batch_first=False):
    """ Masks the targets """
    if not batch_first:
         targets = targets.transpose(0,1)
    t = []
    for l, p in zip(targets,sequence_lengths):
        t.append(l[:p].data.tolist())
    return t

n_gpu = torch.cuda.device_count()
device = "cpu"

args = Config()

# *** LOAD DATA ***
train, dev, test, label_to_id = get_dataloaders(args)
# train, label_to_id = get_dataloaders(args)
args.ntags = len(label_to_id)

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

trigger_list = ["Life:Be-Born", "Life:Die", "Life:Marry", "Life:Divorce", "Life:Injure"]

id_to_label = dict()
for i,label in enumerate(label_to_id.keys()):
    id_to_label[i] = label

elmo = Elmo(args.options_file, args.weight_file, 2, dropout=0)

for epoch in trange(args.num_epochs):
    train_loss = 0
    # total = 0
    # correct = 0
    for step, (sentences, label_ids) in enumerate(train):
        continue
        seq_lengths = [len(sent) for sent in sentences]
        char_ids = batch_to_ids(sentences)
        char_ids.to(device)
        # Returns a dict with keys=["elmo_representations" -> [BxSx1024, BxSx1024], "mask" -> BxS]. in "elmo_representations" returns a list, but two tensors are copies of each other.
        embeddings = elmo(char_ids)
        word_input = embeddings['elmo_representations'][0].detach().requires_grad_(False)
        mask = embeddings['mask'].clone().detach()
        del embeddings
        word_input.to(device)
        mask.to(device)
        loss, _ = model(word_input, mask=mask, labels=label_ids)

        if n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu.

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        print(train_loss / ((step+1)*args.train_batch_size))
        # masked_targets = mask_targets(label_ids, seq_lengths, batch_first=True)

        # t_ = mask.type(torch.LongTensor).sum().item()
        # total += t_
        # c_ = sum([1 if p[i] == mt[i] else 0 for p, mt in zip(preds, masked_targets) for i in range(len(p))])
        # correct += c_


    if args.do_eval:
        for step, (sentences, events, doc_ids) in enumerate(dev):
            seq_lengths = [len(sent) for sent in sentences]
            pdb.set_trace()
            char_ids = batch_to_ids(sentences)
            char_ids.to(device)
            embeddings = elmo(char_ids)
            word_input = embeddings['elmo_representations'][0].detach().requires_grad_(False) # TODO: check dimensions
            mask = embeddings['mask'].clone().detach()
            del embeddings
            word_input.to(device)
            mask.to(device)
            preds = model(word_input, mask=mask)
            preds = [p[:i] for p, i in zip(preds, seq_lengths)] # in test time
            # preds = [p.item() for p in preds[:-1]] + preds[-1]

            # TODO: left here
            # TODO: debug this part
            for el in preds:
                pred_triggers = []
                pred_arguments = []
                prev_label = "O"
                trigger_label = ""

                # trigger selection algorithm for arguments.
                # TODO: Think of a more sophisticated way of doing this
                for label in el:
                    if label != "O":
                        if label[2:] in trigger_list:
                            trigger_label = label
                            break

                for idx,label in enumerate(el):
                    label = id_to_label[label]
                    if "B-" in label:
                        if prev_label != "O":
                            if prev_label[2:] in trigger_list:
                                pred_triggers.append(((start_idx, idx), prev_label[2:]))
                            else:
                                if trigger_label:
                                    pred_arguments.append(((start_idx, idx), trigger_label, prev_label[2:]))

                        start_idx = idx

                    elif "I-" in label:
                        if prev_label == "O":
                            start_idx = idx
                        if label[2:] != prev_label[2:]: # label changed
                            if prev_label[2:] in trigger_list:
                                pred_triggers.append(((start_idx, idx), prev_label[2:]))
                            else:
                                if trigger_label:
                                    pred_arguments.append(((start_idx, idx), trigger_label, prev_label[2:]))

                            start_idx = idx

                    prev_label = label

                # TODO: import comparing functions from event_metrics.py and call them here
                # TODO: we also need compute_f1 function
                # TODO: check todo in data.py concerning normalizing offset indexes

            # masked_targets = mask_targets(label_ids, seq_lengths, batch_first=True)
            # t_ = mask.type(torch.LongTensor).sum().item()
            # total += t_
            # c_ = sum([1 if p[i] == mt[i] else 0 for p, mt in zip(preds, masked_targets) for i in range(len(p))])
            # correct += c_
