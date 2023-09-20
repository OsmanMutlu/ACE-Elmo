from collections import Counter
# import pdb
# from dygie.training.f1 import compute_f1

"""
Got this script from https://github.com/dwadden/dygiepp/blob/master/dygie/training/event_metrics.py
Adjusted it to serve our purposes
"""

def score_triggers(predicted_triggers, gold_triggers):
    matched_trigger_ids, matched_trigger_classes = 0, 0
    len_gold_triggers = len(gold_triggers)
    len_predicted_triggers = len(predicted_triggers)
    # if len_gold_triggers > 0:
    #     pdb.set_trace()
    for (token_ix, label) in predicted_triggers:
        # Check whether the offsets match, and whether the labels match.
        for (gold_token_ix, gold_label) in gold_triggers:
            if token_ix == gold_token_ix:
                matched_trigger_ids += 1
                if label == gold_label:
                    matched_trigger_classes += 1

    return len_gold_triggers, len_predicted_triggers, matched_trigger_ids, matched_trigger_classes

def score_arguments(predicted_arguments, gold_arguments):
    matched_argument_ids, matched_argument_classes = 0, 0
    len_gold_arguments = len(gold_arguments)
    len_predicted_arguments = len(predicted_arguments)
    # if len_gold_arguments > 0:
    #     pdb.set_trace()
    for prediction in predicted_arguments:
        ix, trigger_label, arg = prediction
        gold_id_matches = {entry for entry in gold_arguments
                           if entry[0] == ix
                           and entry[1] == trigger_label}
        if gold_id_matches:
            matched_argument_ids += 1
            gold_class_matches = {entry for entry in gold_id_matches if entry[2] == arg}
            if gold_class_matches:
                matched_argument_classes += 1

    return len_gold_arguments, len_predicted_arguments, matched_argument_ids, matched_argument_classes
