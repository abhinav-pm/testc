import argparse
import logging
import os
import re
import collections
import json
from transformers import BertTokenizer
import conll
import util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

def skip_doc(doc_key):
    return False

def normalize_word(word, language):
    if language == "arabic":
        word = word[:word.find("#")]
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word

class DocumentState(object):
    def __init__(self, key):
        self.doc_key = key
        self.tokens = []
        self.subtokens = []
        self.subtoken_map = []
        self.token_end = []
        self.sentence_end = []
        self.info = []
        self.segments = []
        self.segment_subtoken_map = []
        self.segment_info = []
        self.speakers = []
        self.pronouns = []
        self.clusters = collections.defaultdict(list)
        self.coref_stacks = collections.defaultdict(list)

    def finalize(self):
        # Populate speakers for the segments
        for segment, subtoken_map, info in zip(self.segments, self.segment_subtoken_map, self.segment_info):
            speakers = []
            for i, _ in enumerate(segment):
                if i == 0 or i == len(segment) - 1:
                    speakers.append('[SPL]')
                else:
                    if info[i] is not None and len(info[i]) > 9:
                        speakers.append(info[i][9])
                    else:
                        speakers.append('[SPL]')
            self.speakers.append(speakers)

        # Extract clusters
        first_subtoken_idx = 0
        subtokens_info = util.flatten(self.segment_info)
        while first_subtoken_idx < len(subtokens_info):
            subtoken_info = subtokens_info[first_subtoken_idx]
            coref = subtoken_info[-2] if subtoken_info is not None else '-'
            if coref != '-':
                last_subtoken_idx = first_subtoken_idx + subtoken_info[-1] - 1
                for part in coref.split('|'):
                    if part[0] == '(':
                        if part[-1] == ')':
                            cluster_id = part[1:-1]
                            if cluster_id:
                                try:
                                    cluster_id = int(cluster_id)
                                    self.clusters[cluster_id].append((first_subtoken_idx, last_subtoken_idx))
                                except ValueError:
                                    logger.warning(f"Invalid cluster ID: {cluster_id}")
                        else:
                            cluster_id = part[1:]
                            if cluster_id:
                                try:
                                    cluster_id = int(cluster_id)
                                    self.coref_stacks[cluster_id].append(first_subtoken_idx)
                                except ValueError:
                                    logger.warning(f"Invalid cluster ID: {cluster_id}")
                    elif part[-1] == ')':
                        cluster_id = part[:-1]
                        if cluster_id:
                            try:
                                cluster_id = int(cluster_id)
                                start = self.coref_stacks[cluster_id].pop()
                                self.clusters[cluster_id].append((start, last_subtoken_idx))
                            except ValueError:
                                logger.warning(f"Invalid cluster ID: {cluster_id}")
            first_subtoken_idx += 1

        # Merge clusters
        merged_clusters = []
        for cluster in self.clusters.values():
            existing = None
            for mention in cluster:
                for merged_cluster in merged_clusters:
                    if mention in merged_cluster:
                        existing = merged_cluster
                        break
                if existing is not None:
                    break
            if existing is not None:
                logger.info("Merging clusters")
                existing.update(cluster)
            else:
                merged_clusters.append(set(cluster))

        merged_clusters = [list(cluster) for cluster in merged_clusters]
        
        # Generate sentence map
        sentence_map = []
        sent_idx = 0
        for segment in self.segments:
            sentence_map.append(sent_idx)  # [CLS]
            seg_len = len(segment) - 2  # Subtract 2 for [CLS] and [SEP]
            for i in range(seg_len):
                sentence_map.append(sent_idx)
                flat_idx = sum(len(seg) - 2 for seg in self.segments[:self.segments.index(segment)]) + i
                if flat_idx < len(self.sentence_end) and self.sentence_end[flat_idx]:
                    sent_idx += 1
            sentence_map.append(sent_idx)  # [SEP]

        # Sanity checks
        assert len(util.flatten(self.segments)) == len(util.flatten(self.speakers))
        subtoken_map = util.flatten(self.segment_subtoken_map)
        assert len(util.flatten(self.segments)) == len(subtoken_map)
        assert len(util.flatten(self.segments)) == len(sentence_map)

        return {
            "doc_key": self.doc_key,
            "tokens": self.tokens,
            "sentences": self.segments,
            "speakers": self.speakers,
            "constituents": [],
            "ner": [],
            "clusters": merged_clusters,
            'sentence_map': sentence_map,
            "subtoken_map": subtoken_map,
            'pronouns': self.pronouns
        }

def split_into_segments(document_state, max_seg_len, constraints1, constraints2, tokenizer):
    curr_idx = 0
    prev_token_idx = 0
    while curr_idx < len(document_state.subtokens):
        end_idx = min(curr_idx + max_seg_len - 1 - 2, len(document_state.subtokens) - 1)
        while end_idx >= curr_idx and not constraints1[end_idx]:
            end_idx -= 1
        if end_idx < curr_idx:
            logger.info(f'{document_state.doc_key}: no sentence end found; trying token end')
            end_idx = min(curr_idx + max_seg_len - 1 - 2, len(document_state.subtokens) - 1)
            while end_idx >= curr_idx and not constraints2[end_idx]:
                end_idx -= 1
            if end_idx < curr_idx:
                end_idx = min(curr_idx + max_seg_len - 1 - 2, len(document_state.subtokens) - 1)

        if curr_idx >= len(document_state.subtokens):
            break

        segment = [tokenizer.cls_token] + document_state.subtokens[curr_idx:end_idx + 1] + [tokenizer.sep_token]
        document_state.segments.append(segment)

        subtoken_map = document_state.subtoken_map[curr_idx:end_idx + 1]
        document_state.segment_subtoken_map.append([prev_token_idx] + subtoken_map + [subtoken_map[-1]])
        
        document_state.segment_info.append([None] + document_state.info[curr_idx:end_idx + 1] + [None])
        
        curr_idx = end_idx + 1
        prev_token_idx = subtoken_map[-1]

def get_document(doc_key, doc_lines, language, seg_len, tokenizer):
    document_state = DocumentState(doc_key)
    word_idx = -1

    current_sentence = []
    for line in doc_lines:
        row = line.split()
        if len(row) == 0:
            if current_sentence:
                document_state.sentence_end[-1] = True
                current_sentence = []
        else:
            assert len(row) >= 12
            word_idx += 1
            word = normalize_word(row[3], language)
            subtokens = tokenizer.tokenize(word)
            document_state.tokens.append(word)
            current_sentence.append(word)
            
            document_state.token_end += [False] * (len(subtokens) - 1) + [True]
            
            for idx, subtoken in enumerate(subtokens):
                document_state.subtokens.append(subtoken)
                info = None if idx != 0 else (row + [len(subtokens)])
                document_state.info.append(info)
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(word_idx)
            
            # Check if this token ends the sentence
            if any(word.endswith(p) for p in ['.', '?', '!']):
                document_state.sentence_end[-1] = True
                current_sentence = []

    # Handle the last sentence
    if current_sentence:
        document_state.sentence_end[-1] = True

    constraints1 = document_state.sentence_end if language != 'arabic' else document_state.token_end
    split_into_segments(document_state, seg_len, constraints1, document_state.token_end, tokenizer)
    return document_state.finalize()

def read_file_with_fallback_encodings(file_path):
    encodings = ['utf-8', 'utf-16', 'utf-16le', 'utf-16be']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
    raise Exception(f"Could not decode file {file_path} with any of the attempted encodings")

def minimize_partition(partition, extension, args, tokenizer):
    input_path = os.path.join(args.input_dir, f'{partition}.{args.language}.{extension}')
    output_path = os.path.join(args.output_dir, f'{partition}.{args.language}.{args.seg_len}.jsonlines')
    doc_count = 0
    logger.info(f'Minimizing {input_path}...')

    # Read documents
    documents = []
    try:
        lines = read_file_with_fallback_encodings(input_path)
        current_doc = None
        for line in lines:
            begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
            if begin_document_match:
                doc_key = conll.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
                current_doc = (doc_key, [])
                documents.append(current_doc)
            elif line.startswith('#end document'):
                current_doc = None
            elif current_doc is not None:
                current_doc[1].append(line)
    except Exception as e:
        logger.error(f"Error reading file {input_path}: {str(e)}")
        return

    # Write documents
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for doc_key, doc_lines in documents:
            if skip_doc(doc_key):
                continue
            try:
                document = get_document(doc_key, doc_lines, args.language, args.seg_len, tokenizer)
                output_file.write(json.dumps(document, ensure_ascii=False))
                output_file.write('\n')
                doc_count += 1
            except Exception as e:
                logger.error(f"Error processing document {doc_key}: {str(e)}")

    logger.info(f'Processed {doc_count} documents to {output_path}')

def minimize_language(args):
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)
    minimize_partition('dev', 'v4_gold_conll', args, tokenizer)
    minimize_partition('test', 'v4_gold_conll', args, tokenizer)
    minimize_partition('train', 'v4_gold_conll', args, tokenizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', type=str, default='google-bert/bert-base-multilingual-cased')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seg_len', type=int, default=128)
    parser.add_argument('--language', type=str, default='telugu')

    args = parser.parse_args()
    logger.info(args)
    os.makedirs(args.output_dir, exist_ok=True)

    minimize_language(args)