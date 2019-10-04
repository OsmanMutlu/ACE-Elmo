from collections import Counter

# from dygie.training.f1 import compute_f1

"""
Got this script from https://github.com/dwadden/dygiepp/blob/master/dygie/training/event_metrics.py
Adjusted it to serve our purposes
"""


# Might need trigger id in gold and pred arguments.
def invert_arguments(arguments, triggers):
    """
    For scoring the argument, we don't need the trigger spans to match exactly. We just need the
    trigger label corresponding to the predicted trigger span to be correct.
    """
    # Can't use a dict because multiple triggers could share the same argument.
    inverted = set()
    for k, v in arguments.items():
        if k[0] in triggers:  # If it's not, the trigger this arg points to is null. TODO(dwadden) check.
            trigger_label = triggers[k[0]]
            to_append = (k[1], trigger_label, v)
            inverted.add(to_append)

    return inverted

def score_triggers(predicted_triggers, gold_triggers):
    gold_triggers += len(gold_triggers)
    predicted_triggers += len(predicted_triggers)
    for (token_ix, label) in predicted_triggers:
        # Check whether the offsets match, and whether the labels match.
        for (gold_token_ix, gold_label) in gold_triggers:
            if token_ix == gold_token_ix:
                matched_trigger_ids += 1
                if label == gold_label:
                    matched_trigger_classes += 1

def score_arguments(predicted_arguments, gold_arguments):
    gold_arguments += len(gold_arguments)
    predicted_arguments += len(predicted_arguments)
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

# TODO: fix this function.
def calc_and_print(predicted_events_list, metadata_list):
    for predicted_events, metadata in zip(predicted_events_list, metadata_list):
        # Trigger scoring.
        predicted_triggers = predicted_events["trigger_dict"]
        gold_triggers = metadata["trigger_dict"]
        # TODO: gold_triggers are a list of ((start,end), label)
        score_triggers(predicted_triggers, gold_triggers)

        # Argument scoring.
        predicted_arguments = invert_arguments(predicted_events["argument_dict"], predicted_triggers)
        gold_arguments = invert_arguments(metadata["argument_dict"], gold_triggers)

        # TODO: Arguments are lists of ((start,end), entity_id, arg_label). Entity_ids can be 1,2,3 etc. because we calculate in sentence.
        score_arguments(predicted_arguments, gold_arguments)
