from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer
import re
from lxml import etree
import ipdb
import pandas as pd
import copy

class ACEParser:
    def __init__(self):
        self.sent_tokenizer = PunktSentenceTokenizer()
        # self.word_tokenizer = RegexpTokenizer('\w+|\S+')
        self.word_tokenizer = WhitespaceTokenizer()
        self.root = None
        self.sentence_offsets = []
        self.df = pd.DataFrame(columns=["doc_id", "sentence", "tokens", "events", "entities"])

    def get_text(self, sgm_file):
        with open(sgm_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Gets rid of lines with only tags
        text = re.sub(r"<(.|\s|\n)*?>", r"", text)
        sentence_offsets = list(self.sent_tokenizer.span_tokenize(text))
        sentences = []
        for offset in sentence_offsets:
            sentence_text = text[offset[0]:offset[1]]
            sentences.append(sentence_text)

        self.sentence_offsets = sentence_offsets
        return text

    def create_tree(self, apf_file):
        with open(apf_file, "r", encoding="utf-8") as f:
            xml_text = f.read()

        root = etree.fromstring(xml_text)
        self.root = root

    def get_extents(self):
        extent_nodes = self.root.xpath("//extent/charseq")
        return [self.get_offset_tuple(extent_node) for extent_node in extent_nodes]

    def get_offset_tuple(self, extent_node):
        return (int(extent_node.get("START")), int(extent_node.get("END"))+1) # +1 makes them exclusive

    def get_sentences(self):
        sentences = []
        for offset in self.sentence_offsets:
            sentence_text = text[offset[0]:offset[1]]
            sentences.append(sentence_text)

        return sentences

    def find_sentence_index(self, offset):

        for i, sent_offset in enumerate(self.sentence_offsets):
             if offset[0] >= sent_offset[0] and offset[1] <= sent_offset[1]:
                 return i

    def offset_to_token(self, start, end, token_offsets, normalize=0):
        # normalize is making start and end relatable to token_offsets
        start -= normalize
        end -= normalize

        # TODO: change this to if end == offset[1]. In the case that end < offset[1] use startswith and extend token_offsets list
        for i, offset in enumerate(token_offsets):
            if end <= offset[1]:
                for j in range(i,-1,-1):
                    if start >= token_offsets[j][0]:
                        return j, i+1 # Make it exclusive

        raise Exception("Error while converting offset to token indexes. Start offset : %d , End offset : %d Norm : %d, Token offsets : %s" %(start, end, normalize, str(token_offsets)))

    def create_json_output(self, doc_text, filename):
        # doc_id = self.root.xpath("document")[0].get("DOCID")
        doc_id = filename
        event_nodes = self.root.xpath("//event")

        # TODO: We lose coreference information doing it this way. For now it is ok, but need to accomodate the other way too !!!
        event_mentions = []
        for event_node in event_nodes:
            event_type = event_node.get("TYPE")
            event_subtype = event_node.get("SUBTYPE")
            event_id = event_node.get("ID")
            event_mention_nodes = event_node.xpath("event_mention")
            for mention_node in event_mention_nodes:
                # You actually don't need these two for finding which sentence we are talking about.
                # Because we already made sure that all of our extents are covered by sentence offsets.
                # extent_node = mention.xpath("/extent/charseq")[0]
                # extent = get_offset_tuple(extent_node)

                trigger_offset = self.get_offset_tuple(mention_node.xpath("anchor/charseq")[0])

                # find which sentence this belongs. Only need to do this once.
                sent_idx = self.find_sentence_index(trigger_offset)

                event_arguments = []
                arguments = mention_node.xpath("event_mention_argument")
                for argument in arguments:
                    arg_role = argument.get("ROLE")
                    arg_offset = self.get_offset_tuple(argument.xpath("extent/charseq")[0])
                    # TODO: NEED TO ADD ENTITY TYPES, getting them from refids !!!
                    event_arguments.append({"role":arg_role, "start":arg_offset[0], "end":arg_offset[1]})

                event_mentions.append({"event_id":event_id, "event_type":event_type, "event_subtype":event_subtype,
                                      "trigger":{"start":trigger_offset[0], "end":trigger_offset[1]},
                                       "arguments":event_arguments, "sent_idx":sent_idx})

        # For printing later
        # old_event_mentions = copy.deepcopy(event_mentions)

        tokens_list_for_printing = []
        for i,sentence_offset in enumerate(self.sentence_offsets):
            sentence_text = doc_text[sentence_offset[0]:sentence_offset[1]]
            token_offsets = list(self.word_tokenizer.span_tokenize(sentence_text))
            tokens = [sentence_text[offset[0]:offset[1]] for offset in token_offsets]
            tokens_list_for_printing.append(tokens)
            entity_mentions = []
            curr_event_mentions = []

            for j in range(len(event_mentions)):
                mention = event_mentions[j]
                if mention["sent_idx"] == i:
                    # ipdb.set_trace()
                    start_idx, end_idx = self.offset_to_token(mention["trigger"]["start"], mention["trigger"]["end"], token_offsets, normalize=sentence_offset[0])
                    event_mentions[j]["trigger"]["start"] = start_idx
                    event_mentions[j]["trigger"]["end"] = end_idx

                    for k, argument in enumerate(mention["arguments"]):
                        start_idx, end_idx = self.offset_to_token(argument["start"], argument["end"], token_offsets, normalize=sentence_offset[0])
                        event_mentions[j]["arguments"][k]["start"] = start_idx
                        event_mentions[j]["arguments"][k]["end"] = end_idx

                    curr_event_mentions.append(event_mentions[j])

            self.df = self.df.append({"doc_id":doc_id, "sentence":sentence_text,
                                      "tokens":tokens, "events":curr_event_mentions,
                                      "entities":entity_mentions}, ignore_index=True)

        # Printing stuff
        # for mention, old_mention in zip(event_mentions, old_event_mentions):
        #     tokens = tokens_list_for_printing[mention["sent_idx"]]
        #     print("Offset version trigger : %s , Tokens version trigger : %s" %(doc_text[old_mention["trigger"]["start"]:old_mention["trigger"]["end"]], tokens[mention["trigger"]["start"]:mention["trigger"]["end"]]))
        #     for argument, old_argument in zip(mention["arguments"], old_mention["arguments"]):
        #         print("Offset version argument : %s , Tokens version argument : %s" %(doc_text[old_argument["start"]:old_argument["end"]], tokens[argument["start"]:argument["end"]]))

        #     print("===========")

    # TODO: Remove debug stuff
    def fix_offsets(self, extents):
        offsets = self.sentence_offsets
        assert(len(offsets) > 1)
        # print(offsets)
        # print("*************")

        after_count = 0
        before_count = 0
        for extent in extents:
            # Check stuff for printing
            if len([offset for offset in offsets if extent[0] >= offset[0] and extent[1] <= offset[1]]) == 0:
                before_count += 1

            if extent[1] <= offsets[0][1]:
                continue

            for idx in range(1,len(offsets)):
                offset = offsets[idx]
                if extent[1] <= offset[1]: # Ends before this sentence.
                    if extent[0] < offset[0]: # Starts before this sentence
                        # Fixing
                        # print("-------")
                        # print(extent)
                        # print(offsets)
                        for j in range(idx-1,-1,-1): # For all sentences' offsets before this offset
                            del offsets[j+1]
                            if extent[0] >= offsets[j][0]:
                                offsets[j] = (offsets[j][0], offset[1])
                                break

                        # print(offsets)
                        break

                    else: # Nothing wrong with this extent
                        break

            # Check stuff for printing
            if len([offset for offset in offsets if extent[0] >= offset[0] and extent[1] <= offset[1]]) == 0:
                ipdb.set_trace()
                # MISSES some due to spaces between sentences
                # print(extent)
                # print(text[extent[0]:extent[1]])
                after_count += 1

        # print("Before : %d -> After : %d" %(before_count, after_count))
        # print("================================================================================================================")

        self.sentence_offsets = offsets

asd = ACEParser()

with open("/home/osman/ace2005-preprocessing/dev_split.csv", "r") as f:
    dev_split = f.read().splitlines()

after_total_count = 0
total_count = 0
path = "/home/osman/ace_2005_td_v7/data/English/"
for filename in dev_split:
    asd.create_tree(path + filename + ".apf.xml")
    extents = asd.get_extents()
    text = asd.get_text(path + filename + ".sgm")

    asd.fix_offsets(extents)
    # sentences = asd.get_sentences() # TODO: NO need for this

    asd.create_json_output(text, filename)

    print("------------------- Finished one file! -------------------")

asd.df.to_json("dev.json", orient="records", lines=True, force_ascii=False)
    # ******* LOOK AT TENSES ON EVENTS, IGNORE FUTURE EVENTS ******* -> Actually doesn't matter, because we are comparing models, so we use same data.
