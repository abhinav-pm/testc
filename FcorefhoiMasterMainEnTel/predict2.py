import json
import argparse
import logging
import torch
from indicnlp.tokenize import sentence_tokenize, indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from preprocess import get_document
from tensorize import CorefDataProcessor
from run import Runner
import util

logging.getLogger().setLevel(logging.CRITICAL)

def create_telugu_tokenizer():
    normalizer = IndicNormalizerFactory().get_normalizer("te")
    return normalizer

def get_document_from_string(string, seg_len, bert_tokenizer, telugu_tokenizer, genre='nw'):
    doc_key = genre
    doc_lines = []

    normalized_text = telugu_tokenizer.normalize(string)
    sentences = sentence_tokenize.sentence_split(normalized_text, lang='te')
    
    for sent in sentences:
        tokens = indic_tokenize.trivial_tokenize(sent, lang='te')
        for token in tokens:
            cols = [genre] + ['-'] * 11
            cols[3] = token
            doc_lines.append('\t'.join(cols))
        doc_lines.append('\n')

    doc = get_document(doc_key, doc_lines, 'telugu', seg_len, bert_tokenizer)
    return doc

def print_tensor_details(tensor, name):
    if isinstance(tensor, torch.Tensor):
        print(f"{name} type: {type(tensor)}")
        print(f"{name} shape: {tensor.shape}")
        print(f"{name} dtype: {tensor.dtype}")
        print(f"{name} device: {tensor.device}")
        if tensor.dim() == 0:
            print(f"{name} value: {tensor.item()}")
        else:
            print(f"{name} first few elements: {tensor[:5]}")
    else:
        print(f"{name} type: {type(tensor)}")
        print(f"{name} value: {tensor}")
    print()

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

    model.to(model.device)
    telugu_tokenizer = create_telugu_tokenizer()

    while True:
        input_str = str(input('Input document (in telugu):'))
        bert_tokenizer = data_processor.tokenizer
        doc = get_document_from_string(input_str, args.seg_len, bert_tokenizer, telugu_tokenizer)
        tensor_examples, stored_info = data_processor.get_tensor_examples_from_custom_input([doc])
        
        print("Structure of tensor_examples:")
        print(f"Length of tensor_examples: {len(tensor_examples)}")
        print(f"Type of first element: {type(tensor_examples[0])}")
        print(f"Length of first element: {len(tensor_examples[0])}")
        
        # Unpack the tensor examples
        doc_key, example = tensor_examples[0]
        print(f"doc_key: {doc_key}")
        print(f"Number of tensors in example: {len(example)}")
        
        for i, tensor in enumerate(example):
            print_tensor_details(tensor, f"Tensor {i}")
        
        # Run the model
        model.eval()
        with torch.no_grad():
            example_gpu = [d.to(model.device) for d in example[:7]]  # Strip out gold
            outputs = model(*example_gpu)
        
        print("Model outputs:")
        for i, output in enumerate(outputs):
            print_tensor_details(output, f"Output {i}")
        
        predicted_clusters, predicted_spans, predicted_antecedents = runner.predict(model, tensor_examples)

        subtokens = util.flatten(doc['sentences'])
        print('---Predicted clusters:')
        for cluster in predicted_clusters[0]:
            mentions_str = [' '.join(subtokens[m[0]:m[1]+1]) for m in cluster]
            mentions_str = [m.replace(' ##', '') for m in mentions_str]
            mentions_str = [m.replace('##', '') for m in mentions_str]
            print(mentions_str)  # Print out strings
            print(cluster)  # Print out indices
        
        print("\nPredicted spans:")
        print(predicted_spans)
        
        print("\nPredicted antecedents:")
        print(predicted_antecedents)