import json
import argparse
import logging
import os
from indicnlp.tokenize import sentence_tokenize, indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from preprocess import get_document
import util
from tensorize import CorefDataProcessor
from run import Runner

logging.getLogger().setLevel(logging.CRITICAL)

#to run this file use this command
# python predictOutSideFileOutputs.py --config_name best --model_identifier Nov24_05-33-10 --gpu_id 0 --input_folder "D:\IIIT_H\Task3\sampledata\newsampledata\tl_3rdset_xmlTeluguCoref\processed_output" --output_file "D:\IIIT_H2\all_translation2\telugutrans2\Hindi_to_Tel\F_coref-hoi-master_main_Tel\Outs\predictions_output.txt"

def create_telugu_tokenizer():
    normalizer = IndicNormalizerFactory().get_normalizer("te")
    return normalizer

def get_document_from_string(string, seg_len, bert_tokenizer, telugu_tokenizer, genre='nw'):
    doc_key = genre
    doc_lines = []

    normalized_text = telugu_tokenizer.normalize(string)
    sentences = sentence_tokenize.sentence_split(normalized_text, lang='te')
    
    for sentence in sentences:
        tokens = indic_tokenize.trivial_tokenize(sentence, lang='te')
        for token in tokens:
            cols = [genre] + ['-'] * 11
            cols[3] = token
            doc_lines.append('\t'.join(cols))
        doc_lines.append('\n')

    doc = get_document(doc_key, doc_lines, 'telugu', seg_len, bert_tokenizer)
    return doc

def process_file(file_path, model, data_processor, telugu_tokenizer, args, output_file):
    try:
        # Read the input file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Write filename to output
        filename = os.path.basename(file_path)
        output_file.write(f"\n''{filename}''\n")
        output_file.write(f"Input document (in telugu):{text}\n")
        
        # Process the document
        bert_tokenizer = data_processor.tokenizer
        doc = get_document_from_string(text, args.seg_len, bert_tokenizer, telugu_tokenizer)
        tensor_examples, stored_info = data_processor.get_tensor_examples_from_custom_input([doc])
        
        # Get predictions
        predicted_clusters, _, _ = runner.predict(model, tensor_examples)
        
        # Get subtokens
        subtokens = util.flatten(doc['sentences'])
        
        # Write predictions to output
        output_file.write('---Predicted clusters:\n')
        for cluster in predicted_clusters[0]:
            mentions_str = [' '.join(subtokens[m[0]:m[1]+1]) for m in cluster]
            mentions_str = [m.replace(' ##', '') for m in mentions_str]
            mentions_str = [m.replace('##', '') for m in mentions_str]
            output_file.write(str(mentions_str) + '\n')
        
        output_file.write('\n')  # Add blank line between documents
        output_file.flush()  # Ensure writing to file
        
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True,
                        help='Configuration name in experiments.conf')
    parser.add_argument('--model_identifier', type=str, required=True,
                        help='Model identifier to load')
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='GPU id; CPU by default')
    parser.add_argument('--seg_len', type=int, default=512)
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to folder containing input files')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to output file')
    args = parser.parse_args()

    # Initialize model and processors
    runner = Runner(args.config_name, args.gpu_id)
    model = runner.initialize_model(args.model_identifier)
    data_processor = CorefDataProcessor(runner.config)
    telugu_tokenizer = create_telugu_tokenizer()

    # Move model to device
    model.to(model.device)
    model.eval()

    # Get list of input files
    input_files = [f for f in os.listdir(args.input_folder) if f.endswith('_processed.txt')]
    
    # Process all files and write to output
    with open(args.output_file, 'w', encoding='utf-8') as output_file:
        for file_name in input_files:
            file_path = os.path.join(args.input_folder, file_name)
            process_file(file_path, model, data_processor, telugu_tokenizer, args, output_file)
            
    print(f"Finished processing {len(input_files)} files. Results saved to {args.output_file}")