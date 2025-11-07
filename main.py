
import argparse
import os
import logging
import time
import pickle
from tqdm import tqdm
import random


import torch
import numpy as np
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup

from models import TwoStageT5Model, freeze_model_parameters
from data_utils import ABSADataset, IterativeCorrectionDataset, read_line_examples_from_file
from constants import aspect_cate_list
from eval_utils import compute_scores, extract_spans_para

import logging
import os


logger = logging.getLogger(__name__)


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='asqp', type=str, required=True,
                        help="The name of the task, selected from: [asqp, tasd, aste]")
    parser.add_argument("--dataset", default='rest15', type=str, required=True,
                        help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_direct_eval", action='store_true', 
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_inference", action='store_true', 
                        help="Whether to run inference with trained checkpoints")
    parser.add_argument("--custom_prompt", type=str, default=None,
                        help="Custom prompt to add during inference only")
    
    # Two-stage training parameters
    parser.add_argument("--do_correction", action='store_true',
                      help="Run correction (stage 2) training")
    parser.add_argument("--correction_data_path", type=str, default=None,
                      help="Path to correction training data")
    parser.add_argument("--use_first_stage_weights", action='store_true' ,
                      help="Initialize stage 2 with stage 1 weights")
    parser.add_argument("--eval_with_correction", action='store_true',
                      help="Use correction model during evaluation")
    parser.add_argument("--save_steps", type=int, default=50,
                      help="Save checkpoint every N epochs")
    parser.add_argument("--num_correction_epochs", default=10, type=int,
                  help="Number of epochs for correction stage")
    parser.add_argument("--correction_checkpoint", type=str, default="correction_model",
                      help="Correction model checkpoint path")
    parser.add_argument("--save_start_epoch", type=int, default=5,
                      help="Start saving checkpoints from epoch N")
    parser.add_argument("--batch_id", type=str, default="",
                      help="Batch identifier for output files")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--resume_epoch", type=int, default=None,
                        help="Starting epoch for resumed training")
    parser.add_argument("--correction_prompt", type=str, default=None,
                        help="Prompt for stage 2 correction training")

    parser.add_argument("--freeze_encoder", action='store_true',
                    help="Freeze entire encoder")
    parser.add_argument("--freeze_encoder_layers", type=int, default=3,
                    help="Number of encoder layers to freeze from bottom")
    parser.add_argument("--freeze_decoder_layers", type=int, default=6,
                    help="Number of decoder layers to freeze")
    parser.add_argument("--freeze_embeddings", action='store_true',
                    help="Freeze embedding layer")
    parser.add_argument("--freeze_from_bottom", action='store_true',
                    help="Freeze decoder from bottom (default True)")

    # other parameters
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--n_gpu", default=0, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=20, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=666)

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=50, type=float)

    args = parser.parse_args()

    # set up output dir which looks like './outputs/rest15/'
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    output_dir = f"outputs/{args.dataset}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    return args


def get_dataset(tokenizer, type_path, args):
    return ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, 
                       data_type=type_path, max_len=args.max_seq_length)


def train_first_stage(model, train_loader, optimizer, scheduler, device):
    """Train stage 1 (base) model"""
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(train_loader, desc="Stage 1"):
        optimizer.zero_grad()
        
        lm_labels = batch["target_ids"].to(device)
        lm_labels[lm_labels[:, :] == model.tokenizer.pad_token_id] = -100
        
        outputs = model(
            input_ids=batch["source_ids"].to(device),
            attention_mask=batch["source_mask"].to(device),
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'].to(device),
            correction_mode=False
        )
        
        loss = outputs[0]
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        total_loss += loss.item()
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(train_loader)


def train_second_stage(model, train_loader, optimizer, scheduler, device,
                      correct_weight=2.0, error_weight=1.0):
    """Train stage 2 (correction) model with sample weighting"""
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(train_loader, desc="Stage 2"):
        optimizer.zero_grad()
        
        input_ids = batch["source_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        input_texts = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        target_texts = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in target_ids]
        
        lm_labels = batch["target_ids"].clone().to(device)
        lm_labels[lm_labels[:, :] == model.tokenizer.pad_token_id] = -100
        
        sample_weights = []
        for i, (i_text, t_text) in enumerate(zip(input_texts, target_texts)):
            quads_text = ""
            first_aspect_pos = len(i_text)
            
            for aspect_cat in aspect_cate_list:
                pattern = f"{aspect_cat} is"
                pos = i_text.find(pattern)
                if pos != -1 and pos < first_aspect_pos:
                    first_aspect_pos = pos
            
            the_patterns = []
            for aspect_cat in aspect_cate_list:
                the_patterns.append(f"The {aspect_cat} is")
            
            for pattern in the_patterns:
                pos = i_text.find(pattern)
                if pos != -1 and pos < first_aspect_pos:
                    first_aspect_pos = pos
            
            if first_aspect_pos < len(i_text):
                quads_text = i_text[first_aspect_pos:].strip()
            
            if quads_text.strip() == t_text.strip():
                sample_weights.append(correct_weight)
            else:
                sample_weights.append(error_weight)
        
        sample_weights = torch.tensor(sample_weights, device=device)
        
        weighted_loss = model.compute_weighted_loss(
            input_ids=batch["source_ids"].to(device),
            attention_mask=batch["source_mask"].to(device),
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'].to(device),
            sample_weights=sample_weights,
            correction_mode=True
        )
        
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.correction_model.parameters(), max_norm=1.0)
        
        total_loss += weighted_loss.item()
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(train_loader)


def validate_model(model, val_loader, device, correction_mode=False):
    """Validate model performance"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            lm_labels = batch["target_ids"].to(device)
            lm_labels[lm_labels[:, :] == model.tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=batch["source_ids"].to(device),
                attention_mask=batch["source_mask"].to(device),
                labels=lm_labels,
                decoder_attention_mask=batch['target_mask'].to(device),
                correction_mode=correction_mode
            )

            loss = outputs[0]
            total_loss += loss.item()

    return total_loss / len(val_loader)


def evaluate_two_stage(data_loader, model, sents, device, use_correction=False, args=None):
    """Evaluate two-stage model performance"""
    model.eval()
    first_stage_outputs = []
    final_outputs = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            first_stage_outs = model.model.generate(
                input_ids=batch['source_ids'].to(device),
                attention_mask=batch['source_mask'].to(device),
                max_length=128
            )
            
            first_stage_decoded = [model.tokenizer.decode(ids, skip_special_tokens=True) 
                                for ids in first_stage_outs]
            first_stage_outputs.extend(first_stage_decoded)
            
            if use_correction and hasattr(model, 'correction_model') and model.correction_model is not None:
                original_texts = [model.tokenizer.decode(ids, skip_special_tokens=True) 
                               for ids in batch['source_ids']]
                
                outs = model.generate_with_correction(
                    input_ids=batch['source_ids'].to(device),
                    attention_mask=batch['source_mask'].to(device),
                    original_texts=original_texts,
                    max_length=128,
                    num_beams=3
                )
                
                corrected_decoded = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
                final_outputs.extend(corrected_decoded)
            else:
                final_outputs = first_stage_outputs
            
            target = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]
            targets.extend(target)
    
    scores, all_labels, all_preds = compute_scores(final_outputs, targets, sents)
    
    return first_stage_outputs, final_outputs, targets, scores


def evaluate_original(data_loader, model, sents, device, custom_prompt=None):
    """Evaluate original single-stage model"""
    model.eval()
    outputs, targets = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            if custom_prompt:
                original_inputs = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["source_ids"]]
                
                modified_inputs = []
                for orig in original_inputs:
                    try:
                        parts = orig.split("Input:")
                        prompt = parts[0]
                        examples = ""
                        if len(parts) > 2:
                            examples = "Input:" + "Input:".join(parts[1:-1])
                        
                        # Extract input text after last "Input:" and before "Output:"
                        input_text = parts[-1].split("Output:")[0].strip()
                        # Alternatively, if prompt is independent of the actual question text
                        modified_input = f"{prompt}{custom_prompt}{examples}\nInput: {input_text} Output:"
                        
                        modified_inputs.append(modified_input)
                    except:
                        modified_inputs.append(orig)
                
                # Re-encode modified inputs
                modified_encodings = model.tokenizer.batch_encode_plus(
                    modified_inputs, 
                    max_length=512,
                    padding="max_length", 
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Inference with modified input
                outs = model.model.generate(
                    input_ids=modified_encodings['input_ids'].to(device), 
                    attention_mask=modified_encodings['attention_mask'].to(device),
                    max_length=128
                )
            else:
                # Inference with original input
                outs = model.model.generate(
                    input_ids=batch['source_ids'].to(device), 
                    attention_mask=batch['source_mask'].to(device), 
                    max_length=128
                )

            dec = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            outputs.extend(dec)
            
            # Get target output
            target = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]
            targets.extend(target)

    # Compute scores
    scores, all_labels, all_preds = compute_scores(outputs, targets, sents)
    return scores


# Extract epoch from checkpoint path
def extract_epoch_from_checkpoint(checkpoint_path):
    """
    Extract starting epoch number from checkpoint path
    Format example: correction_checkpoint_epoch_10
    """
    try:
        # Extract last number as epoch
        parts = os.path.basename(checkpoint_path).split('_')
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        
        # If no number found, try matching specific format
        if "epoch" in checkpoint_path.lower():
            epoch_str = checkpoint_path.split("epoch_")[-1].split("_")[0].split("/")[0]
            return int(epoch_str)
    except:
        pass
    
    # Default return 0
    return 0


class LoggingCallback:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def on_validation_end(self, val_loss, epoch):
        self.logger.info(f"***** Validation results after epoch {epoch} *****")
        self.logger.info(f"Validation Loss: {val_loss:.4f}")

    def on_test_end(self, metrics):
        self.logger.info("***** Test results *****")
        
        output_test_results_file = os.path.join(self.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            # Handle different types of metrics
            if isinstance(metrics, (tuple, list)):
                metrics = metrics[0]  # Get first element
            
            # Convert metrics to dictionary if it's not already
            if not isinstance(metrics, dict):
                metrics = {'score': metrics}
            
            # Write metrics
            for key in sorted(metrics.keys()):
                if key not in ["log", "progress_bar"]:
                    self.logger.info(f"{key} = {metrics[key]}")
                    writer.write(f"{key} = {metrics[key]}\n")


# ========== Main Program Start ==========

# initialization
args = init_args()
print("\n", "-"*30, f"NEW EXP: ASQP on {args.dataset}", "-"*30, "\n")

# set random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
print(f"Random seed set to {args.seed}")

# sanity check
# show one sample to check the code and the expected output
tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Here is an example (from the dev set):")
dev_dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, 
                      data_type='dev', max_len=args.max_seq_length)
data_sample = dev_dataset[7]  # a random data sample
print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))

train_dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, 
                      data_type='train', max_len=args.max_seq_length)


# Initialization of the LoggingCallback
logging_callback = LoggingCallback(args.output_dir)

if args.do_train:
    print("\n====== Stage 1 Training ======")
    
    base_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    two_stage_model = TwoStageT5Model(args, base_model, tokenizer)
    two_stage_model = two_stage_model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                          drop_last=True, shuffle=True, num_workers=4)
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in two_stage_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in two_stage_model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = (len(train_loader.dataset) // (args.train_batch_size * max(1, args.n_gpu))) // args.gradient_accumulation_steps * float(args.num_train_epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    
    for epoch in range(int(args.num_train_epochs)):
        train_loss = train_first_stage(two_stage_model, train_loader, optimizer, scheduler, device)
        print(f"Epoch {epoch+1}/{args.num_train_epochs}, Train Loss: {train_loss:.4f}")
        
        val_dataset = get_dataset(tokenizer, "dev", args)
        val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, num_workers=4)
        val_loss = validate_model(two_stage_model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.num_train_epochs}, Val Loss: {val_loss:.4f}")
        
        logging_callback.on_validation_end(val_loss, epoch+1)
        
        if (epoch + 1) % args.save_steps == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"first_stage_epoch_{epoch+1}")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            two_stage_model.model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint: {checkpoint_dir}")
    
    output_dir_first_stage = os.path.join(args.output_dir, "first_stage")
    if not os.path.exists(output_dir_first_stage):
        os.makedirs(output_dir_first_stage)
    two_stage_model.model.save_pretrained(output_dir_first_stage)
    tokenizer.save_pretrained(output_dir_first_stage)
    
    print("Stage 1 training completed!")

if args.do_correction:
    print("\n====== Stage 2 (Correction) Training ======")
    
    start_epoch = 0
    
    if args.resume_from_checkpoint:
        checkpoint_path = os.path.join(args.output_dir, args.resume_from_checkpoint)
        
        if os.path.exists(checkpoint_path):
            print(f"Resuming from: {checkpoint_path}")
            
            if args.resume_epoch is not None:
                start_epoch = args.resume_epoch
            else:
                start_epoch = extract_epoch_from_checkpoint(checkpoint_path)
            
            correction_model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
            print(f"Loaded correction checkpoint")
        else:
            print(f"Warning: checkpoint not found at {checkpoint_path}")
            print(f"Training from scratch")
            
            if args.use_first_stage_weights:
                first_stage_model_path = os.path.join(args.output_dir, "first_stage")
                print(f"Initializing from stage 1: {first_stage_model_path}")
                correction_model = T5ForConditionalGeneration.from_pretrained(first_stage_model_path)
            else:
                print(f"Initializing from pretrained: {args.model_name_or_path}")
                correction_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    else:
        # Not resuming from checkpoint, training from scratch
        if args.use_first_stage_weights:
            first_stage_model_path = os.path.join(args.output_dir, "first_stage")
            print(f"Initializing correction model from stage 1: {first_stage_model_path}")
            correction_model = T5ForConditionalGeneration.from_pretrained(first_stage_model_path)
        else:
            print(f"Initializing correction model from pretrained: {args.model_name_or_path}")
            correction_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    
    print(f"Loading stage 1 model")
    first_stage_model_path = os.path.join(args.output_dir, "first_stage")
    if not os.path.exists(first_stage_model_path):
        print("Warning: stage 1 model not found")
        first_stage_model_path = args.output_dir
    
    base_model = T5ForConditionalGeneration.from_pretrained(first_stage_model_path)
    two_stage_model = TwoStageT5Model(args, base_model, tokenizer, correction_model, 
                                    correction_prompt=args.correction_prompt)
    two_stage_model = two_stage_model.to(device)
    
    correction_data_path = args.correction_data_path
    if not os.path.exists(correction_data_path):
        print(f"Correction data not found: {correction_data_path}")
        exit(1)
    
    correction_dataset = IterativeCorrectionDataset(
        tokenizer=tokenizer, 
        data_dir=args.dataset,
        data_type=correction_data_path,
        max_len=args.max_seq_length,
        correction_prompt=args.correction_prompt
    )
    if args.correction_prompt:
        print(f"Using prompt: '{args.correction_prompt}'")    

    correction_loader = DataLoader(
        correction_dataset, 
        batch_size=args.train_batch_size,
        drop_last=True, 
        shuffle=True, 
        num_workers=4
    )
    
    if args.freeze_encoder or args.freeze_decoder_layers > 0 or args.freeze_embeddings:
        print("\nFreezing layers...")
        freeze_model_parameters(
            two_stage_model.correction_model, 
            freeze_encoder=args.freeze_encoder,
            freeze_encoder_layers=args.freeze_encoder_layers,
            freeze_decoder_layers=args.freeze_decoder_layers,
            freeze_embeddings=args.freeze_embeddings,
            freeze_from_bottom=args.freeze_from_bottom
        )
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in two_stage_model.correction_model.named_parameters() 
                   if not any(nd in n for nd in ['bias', 'LayerNorm.weight']) and p.requires_grad],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in two_stage_model.correction_model.named_parameters() 
                   if any(nd in n for nd in ['bias', 'LayerNorm.weight'])],
         'weight_decay': 0.0}
    ]
    
    remaining_epochs = int(args.num_correction_epochs)
    print(f"Training epochs: {remaining_epochs}, starting from: {start_epoch}")
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = (len(correction_loader.dataset) // (args.train_batch_size * max(1, args.n_gpu))) // args.gradient_accumulation_steps * float(remaining_epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    
    for epoch in range(remaining_epochs):
        current_epoch = start_epoch + epoch + 1
        
        train_loss = train_second_stage(two_stage_model, correction_loader, optimizer, scheduler, device, correct_weight=1.0, error_weight=1.0)
        print(f"Epoch {current_epoch}/{start_epoch + remaining_epochs}, Train Loss: {train_loss:.4f}")
        
        if ((epoch + 1) % args.save_steps == 0) or (epoch == remaining_epochs - 1):
            checkpoint_dir = os.path.join(args.output_dir, f"correction_checkpoint_epoch_{current_epoch}")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            two_stage_model.correction_model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint: {checkpoint_dir}")
    
    output_dir_correction = os.path.join(args.output_dir, "correction_model")
    if not os.path.exists(output_dir_correction):
        os.makedirs(output_dir_correction)
    two_stage_model.correction_model.save_pretrained(output_dir_correction)
    tokenizer.save_pretrained(output_dir_correction)
    print(f"Stage 2 training completed! Final epoch: {current_epoch}")

if args.do_direct_eval:
    print("\n====== Two-Stage Evaluation ======")

    sents, _ = read_line_examples_from_file(f'data/{args.dataset}/test.txt')

    test_dataset = ABSADataset(tokenizer, data_dir=args.dataset, 
                           data_type='test', max_len=args.max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

    if not 'two_stage_model' in locals() or two_stage_model is None:
        first_stage_path = os.path.join(args.output_dir, "first_stage")
        if not os.path.exists(first_stage_path):
            first_stage_path = args.output_dir
        
        print(f"Loading stage 1: {first_stage_path}")
        base_model = T5ForConditionalGeneration.from_pretrained(first_stage_path)
        
        # Use correction model if needed
        if args.eval_with_correction:
            correction_path = os.path.join(args.output_dir, args.correction_checkpoint)
            if os.path.exists(correction_path):
                print(f"Loading correction model: {correction_path}")
                correction_model = T5ForConditionalGeneration.from_pretrained(correction_path)
                # Create model with correction prompt
                two_stage_model = TwoStageT5Model(
                    args, base_model, tokenizer, correction_model, 
                    correction_prompt=args.correction_prompt
                )
                if args.correction_prompt:
                    print(f"Using correction prompt: '{args.correction_prompt}'")
            else:
                print("Correction model not found, using stage 1 only for evaluation")
                two_stage_model = TwoStageT5Model(args, base_model, tokenizer)
        else:
            two_stage_model = TwoStageT5Model(args, base_model, tokenizer)
        
        two_stage_model = two_stage_model.to(device)

    # Get outputs using improved two-stage evaluation function
    first_stage_outputs, final_outputs, targets, scores = evaluate_two_stage(
        test_loader, two_stage_model, sents, device, use_correction=args.eval_with_correction, args=args
    )
    
    # Record test results
    logging_callback.on_test_end(scores)

    # Write results to file
    log_file_path = f"results_log/{args.dataset}_{'corrected' if args.eval_with_correction else 'base'}.txt"
    local_time = time.asctime(time.localtime(time.time()))

    exp_settings = f"Dataset={args.dataset}; Train bs={args.train_batch_size}, num_epochs={args.num_train_epochs}, seed={args.seed}, corrected={args.eval_with_correction}"
    
    # Use correct scores dictionary
    scores_dict = scores
    
    # Process score output
    if isinstance(scores_dict, dict) and all(key in scores_dict for key in ['precision', 'recall', 'f1']):
        exp_results = f"precision: {scores_dict['precision']:.4f}, recall: {scores_dict['recall']:.4f}, F1 = {scores_dict['f1']:.4f}"
        if all(key in scores_dict for key in ['n_gold', 'n_pred', 'n_tp']):
            exp_results = (f"number of gold spans: {scores_dict['n_gold']}, predicted spans: {scores_dict['n_pred']}, hit: {scores_dict['n_tp']}\n{exp_results}")
    else:
        # Alternative handling
        exp_results = f"Output: {scores_dict}"
    
    log_str = f'============================================================\n'
    log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

    if not os.path.exists('./results_log'):
        os.mkdir('./results_log')

    with open(log_file_path, "a+") as f:
        f.write(log_str)

if args.do_inference:
    print("\n====== Two-Stage Inference ======")

    print(f"Loading trained models from {args.output_dir}")
    print('Note: Pre-trained models required, `do_train` should be False')
    
    first_stage_path = os.path.join(args.output_dir, "first_stage")
    if not os.path.exists(first_stage_path):
        first_stage_path = args.output_dir
    
    if not 'two_stage_model' in locals() or two_stage_model is None:
        tokenizer = T5Tokenizer.from_pretrained(first_stage_path)
        base_model = T5ForConditionalGeneration.from_pretrained(first_stage_path)
        
        if args.eval_with_correction:
            correction_path = os.path.join(args.output_dir, args.correction_checkpoint)
            if os.path.exists(correction_path):
                print(f"Loading correction model: {correction_path}")
                correction_model = T5ForConditionalGeneration.from_pretrained(correction_path)
                two_stage_model = TwoStageT5Model(
                    args, base_model, tokenizer, correction_model, 
                    correction_prompt=args.correction_prompt
                )
                if args.correction_prompt:
                    print(f"Using prompt: '{args.correction_prompt}'")
            else:
                print("Correction model not found, using stage 1 only")
                two_stage_model = TwoStageT5Model(args, base_model, tokenizer)
        else:
            two_stage_model = TwoStageT5Model(args, base_model, tokenizer)
        
        two_stage_model = two_stage_model.to(device)

    sents, _ = read_line_examples_from_file(f'data/{args.dataset}/test.txt')

    test_dataset = ABSADataset(tokenizer, data_dir=args.dataset, 
                           data_type='test', max_len=args.max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

    if args.custom_prompt:
        print(f"\nEvaluation with custom prompt: '{args.custom_prompt}'")
        scores = evaluate_original(test_loader, two_stage_model, sents, device, args.custom_prompt)
    else:
        first_stage_outputs, final_outputs, targets, scores = evaluate_two_stage(
            test_loader, two_stage_model, sents, device, 
            use_correction=args.eval_with_correction
        )
    
    log_file_path = f"results_log/{args.dataset}_inference.txt"
    local_time = time.asctime(time.localtime(time.time()))

    exp_settings = f"Dataset={args.dataset}; corrected={args.eval_with_correction}"
    
    if isinstance(scores, dict) and all(key in scores for key in ['n_gold', 'n_pred', 'n_tp']):
        exp_results = (f"number of gold spans: {scores['n_gold']}, predicted spans: {scores['n_pred']}, hit: {scores['n_tp']}\n"
                      f"precision: {scores['precision']:.4f}, recall: {scores['recall']:.4f}, F1 = {scores['f1']:.4f}")
    else:
        exp_results = f"precision: {scores['precision']:.4f}, recall: {scores['recall']:.4f}, F1 = {scores['f1']:.4f}"
    
    log_str = f'============================================================\n'
    log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

    if not os.path.exists('./results_log'):
        os.mkdir('./results_log')

    with open(log_file_path, "a+") as f:
        f.write(log_str)

