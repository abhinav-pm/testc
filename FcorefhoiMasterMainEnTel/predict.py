import json
import argparse
import logging
from indicnlp.tokenize import sentence_tokenize, indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from preprocess import get_document
import util
from tensorize import CorefDataProcessor
# from tensorizepredict import CorefDataProcessor #use if there is any problem in prediction and change the   max_segment_len = to 512 in conf file
from run import Runner

logging.getLogger().setLevel(logging.CRITICAL)

def create_telugu_tokenizer():
    normalizer = IndicNormalizerFactory().get_normalizer("te")
    return normalizer

def get_document_from_string(string, seg_len, bert_tokenizer, telugu_tokenizer, genre='nw'):
    doc_key = genre  # See genres in experiment config
    doc_lines = []

    # Normalize and tokenize the Telugu text
    normalized_text = telugu_tokenizer.normalize(string)
    sentences = sentence_tokenize.sentence_split(normalized_text, lang='te')
    
    for sentence in sentences:
        tokens = indic_tokenize.trivial_tokenize(sentence, lang='te')
        for token in tokens:
            cols = [genre] + ['-'] * 11
            cols[3] = token
            doc_lines.append('\t'.join(cols))
        doc_lines.append('\n')  # End of sentence

    doc = get_document(doc_key, doc_lines, 'telugu', seg_len, bert_tokenizer)
    return doc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True,
                        help='Configuration name in experiments.conf')
    parser.add_argument('--model_identifier', type=str, required=True,
                        help='Model identifier to load')
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='GPU id; CPU by default')
    parser.add_argument('--seg_len', type=int, default=512)
    parser.add_argument('--jsonlines_path', type=str, default=None,
                        help='Path to custom input from file; input from console by default')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save output')
    args = parser.parse_args()

    runner = Runner(args.config_name, args.gpu_id)
    model = runner.initialize_model(args.model_identifier)
    data_processor = CorefDataProcessor(runner.config)

    if args.jsonlines_path:
        # Input from file
        with open(args.jsonlines_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        docs = [json.loads(line) for line in lines]
        tensor_examples, stored_info = data_processor.get_tensor_examples_from_custom_input(docs)
        predicted_clusters, _, _ = runner.predict(model, tensor_examples)

        if args.output_path:
            with open(args.output_path, 'w', encoding='utf-8') as f:
                for i, doc in enumerate(docs):
                    doc['predicted_clusters'] = predicted_clusters[i]
                    f.write(json.dumps(doc, ensure_ascii=False))
                    f.write('\n')
            print(f'Saved prediction in {args.output_path}')
    else:
        # Interactive input
        model.to(model.device)
        telugu_tokenizer = create_telugu_tokenizer()
        while True:
            input_str = str(input('Input document (in telugu):'))
            bert_tokenizer = data_processor.tokenizer
            doc = get_document_from_string(input_str, args.seg_len, bert_tokenizer, telugu_tokenizer)
            tensor_examples, stored_info = data_processor.get_tensor_examples_from_custom_input([doc])
            predicted_clusters, _, _ = runner.predict(model, tensor_examples)

            subtokens = util.flatten(doc['sentences'])
            print('---Predicted clusters:')
            for cluster in predicted_clusters[0]:
                mentions_str = [' '.join(subtokens[m[0]:m[1]+1]) for m in cluster]
                mentions_str = [m.replace(' ##', '') for m in mentions_str]
                mentions_str = [m.replace('##', '') for m in mentions_str]
                print(mentions_str)  # Print out strings
                # print(cluster)  # Print out indices