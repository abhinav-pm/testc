import json
import argparse
import logging
import torch
import os
from tqdm import tqdm
from indicnlp.tokenize import sentence_tokenize, indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from preprocess import get_document
from tensorize import CorefDataProcessor
from run import Runner
import util
from metrics import CorefEvaluator

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

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

def evaluate_folder(input_folder, runner, data_processor, telugu_tokenizer, args):
    # Initialize evaluator
    evaluator = CorefEvaluator()
    total_documents = 0
    processed_documents = 0

    # Get all text files in the folder
    files = [f for f in os.listdir(input_folder) if f.endswith('_processed.txt')]
    
    logger.info(f"Found {len(files)} documents to process")
    
    # Process each file
    for file_name in tqdm(files, desc="Processing documents"):
        try:
            file_path = os.path.join(input_folder, file_name)
            
            # Read the document
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            # Skip empty documents
            if not text:
                continue

            # Process document
            bert_tokenizer = data_processor.tokenizer
            doc = get_document_from_string(text, args.seg_len, bert_tokenizer, telugu_tokenizer)
            tensor_examples, stored_info = data_processor.get_tensor_examples_from_custom_input([doc])
            
            # Get predictions
            predicted_clusters, predicted_spans, predicted_antecedents = runner.predict(model, tensor_examples)
            
            # Get document subtokens
            subtokens = util.flatten(doc['sentences'])
            
            # Update evaluator
            doc_key = tensor_examples[0][0]
            gold_clusters = stored_info['gold'].get(doc_key, [])  # Empty list if no gold clusters
            
            # Convert predicted clusters to the format expected by the evaluator
            predicted = predicted_clusters[0]  # Get clusters for the first (and only) document
            mention_to_predicted = {(m[0], m[1]): i for i, cluster in enumerate(predicted) 
                                  for m in cluster}
            
            # Update evaluator with this document's predictions
            evaluator.update(predicted, gold_clusters, mention_to_predicted)
            
            processed_documents += 1
            
            # Save predictions if output path is specified
            if args.output_path:
                output_file = os.path.join(args.output_path, f"{file_name}_predictions.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    predictions = {
                        'file_name': file_name,
                        'clusters': [[' '.join(subtokens[m[0]:m[1]+1]).replace(' ##', '').replace('##', '') 
                                    for m in cluster] for cluster in predicted_clusters[0]],
                        'spans': predicted_spans[0],
                        'antecedents': predicted_antecedents[0]
                    }
                    json.dump(predictions, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {str(e)}")
            continue
        
        total_documents += 1

    # Calculate final metrics
    precision, recall, f1 = evaluator.get_prf()
    
    # Print results
    logger.info("=== Final Evaluation Results ===")
    logger.info(f"Total documents processed: {processed_documents}/{total_documents}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1: {f1:.4f}")
    
    return precision, recall, f1

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
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save predictions')
    args = parser.parse_args()

    # Create output directory if specified
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)

    # Initialize model and processors
    runner = Runner(args.config_name, args.gpu_id)
    model = runner.initialize_model(args.model_identifier)
    data_processor = CorefDataProcessor(runner.config)
    telugu_tokenizer = create_telugu_tokenizer()

    # Move model to device
    model.to(model.device)
    model.eval()

    # Evaluate folder
    precision, recall, f1 = evaluate_folder(args.input_folder, runner, data_processor, telugu_tokenizer, args)