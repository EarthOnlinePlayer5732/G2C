# -*- coding: utf-8 -*-

"""
Model Definition Module - Contains definitions and related functions for two-stage T5 model
"""

import torch
from torch import nn
from transformers import T5ForConditionalGeneration


class TwoStageT5Model(nn.Module):
    """
    Two-stage T5 model supporting initial prediction and error correction
    """
    def __init__(self, hparams, base_model, tokenizer, correction_model=None, correction_prompt=None):
        super(TwoStageT5Model, self).__init__()
        self.hparams = hparams
        self.model = base_model
        self.tokenizer = tokenizer
        self.correction_model = correction_model
        self.correction_prompt = correction_prompt
       
        # Ensure tokenizer has [SENTSEP] token
        if '[SENTSEP]' not in tokenizer.get_vocab():
            special_tokens = {'additional_special_tokens': ['[SENTSEP]']}
            tokenizer.add_special_tokens(special_tokens)
           
            # If model is already initialized, need to adjust embedding layer size
            if hasattr(self.model, 'resize_token_embeddings'):
                self.model.resize_token_embeddings(len(tokenizer))
            if self.correction_model and hasattr(self.correction_model, 'resize_token_embeddings'):
                self.correction_model.resize_token_embeddings(len(tokenizer))
    
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None, correction_mode=False):
        """
        Forward propagation, optionally using correction model or base model
        """
        if correction_mode and self.correction_model is not None:
            return self.correction_model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )
        else:
            return self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )
            
    def compute_weighted_loss(self, input_ids, attention_mask, labels, decoder_attention_mask,
                         sample_weights, correction_mode=False):
        """
        Compute weighted loss
       
        Args:
            sample_weights: Weight tensor for each sample [batch_size]
        """
        # Select appropriate model based on correction_mode
        if correction_mode and self.correction_model is not None:
            model_to_use = self.correction_model
        else:
            model_to_use = self.model
       
        # Directly use selected model for forward propagation
        outputs = model_to_use(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True
        )
       
        # Get logits
        logits = outputs.logits
       
        # Calculate loss for each sample and each position
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
       
        # Reshape
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
       
        # Calculate loss at each position
        losses = loss_fct(logits_flat, labels_flat)
        losses = losses.view(labels.size())
       
        # Create mask
        mask = (labels != -100).float()
       
        # Calculate loss for each sample
        sample_losses = (losses * mask).sum(dim=1) / mask.sum(dim=1)
       
        # Apply weights
        weighted_loss = (sample_losses * sample_weights).mean()
       
        return weighted_loss
        
    def generate_with_correction(self, input_ids, attention_mask=None, original_texts=None, **generate_kwargs):
        """
        Two-stage generation: first generate initial output with base model, then correct with correction model
        
        Args:
            input_ids: Input IDs
            attention_mask: Attention mask
            original_texts: List of original texts for correction stage
            generate_kwargs: Generation parameters
            
        Returns:
            corrected_outputs: Corrected outputs
        """
        # Stage 1: Generate initial output
        initial_outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs
        )
        
        if self.correction_model is None:
            return initial_outputs
            
        # Decode initial output
        decoded_outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) 
                          for ids in initial_outputs]
        
        # If original texts not provided, decode directly from input IDs
        if original_texts is None:
            original_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) 
                             for ids in input_ids]
        
        # Create correction inputs
        correction_inputs = []
        for i, (orig_text, decoded_output) in enumerate(zip(original_texts, decoded_outputs)):
            # Use prompt (if available)
            if hasattr(self, 'correction_prompt') and self.correction_prompt:
                correction_input = f"{orig_text} {self.correction_prompt} [SENTSEP] {decoded_output}"
            else:
                correction_input = f"{orig_text} [SENTSEP] {decoded_output}"
            correction_inputs.append(correction_input)
        
        # Tokenize correction inputs
        correction_input_ids = self.tokenizer.batch_encode_plus(
            correction_inputs,
            max_length=input_ids.size(1),
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["input_ids"].to(input_ids.device)
        
        # Stage 2: Correct initial output
        corrected_outputs = self.correction_model.generate(
            input_ids=correction_input_ids,
            attention_mask=None,  # Use default attention mask
            **generate_kwargs
        )
        
        return corrected_outputs


def freeze_model_parameters(model, freeze_encoder=False, freeze_encoder_layers=0, freeze_decoder_layers=0, freeze_embeddings=False, freeze_from_bottom=True):
    """
    Freeze specified parts of T5 model parameters
    
    Args:
        model: T5ForConditionalGeneration model
        freeze_encoder: Whether to freeze entire encoder (backward compatibility)
        freeze_encoder_layers: Number of encoder layers to freeze (from bottom)
        freeze_decoder_layers: Number of decoder layers to freeze
        freeze_embeddings: Whether to freeze token embedding layer
        freeze_from_bottom: Whether to freeze decoder from bottom (True for bottom, False for top)
    """
    # Freeze encoder (two methods: freeze all or freeze by layer)
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif freeze_encoder_layers > 0:
        encoder_blocks = model.encoder.block
        
        # Ensure not exceeding actual number of layers
        n_layers = len(encoder_blocks)
        freeze_encoder_layers = min(freeze_encoder_layers, n_layers)
        
        # Freeze specified number of encoder layers from bottom
        start_idx = 0
        end_idx = freeze_encoder_layers
        
        # Freeze specified layers
        for i in range(start_idx, end_idx):
            for param in encoder_blocks[i].parameters():
                param.requires_grad = False
    
    # Freeze decoder layers
    if freeze_decoder_layers > 0:
        decoder_blocks = model.decoder.block
        
        # Ensure not exceeding actual number of layers
        n_layers = len(decoder_blocks)
        freeze_decoder_layers = min(freeze_decoder_layers, n_layers)
        
        if freeze_from_bottom:
            # Freeze from bottom
            start_idx = 0
            end_idx = freeze_decoder_layers
        else:
            # Freeze from top
            start_idx = n_layers - freeze_decoder_layers
            end_idx = n_layers
        
        # Freeze specified layers
        for i in range(start_idx, end_idx):
            for param in decoder_blocks[i].parameters():
                param.requires_grad = False
    
    # Freeze token embedding layer
    if freeze_embeddings:
        for param in model.shared.parameters():
            param.requires_grad = False
