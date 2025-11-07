# This script contains all data transformation and reading
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Import all constants from constants module to avoid duplication
from constants import aspect_cate_list

# Define sentiment label mappings
senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
senttag2opinion = {'POS': 'great', 'NEG': 'bad', 'NEU': 'ok'}
sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}

# Define prompt templates for different datasets
DATASET_PROMPTS = {
    'rest': """
        Be precise in matching aspects to their correct categories. Use 'it' for implicit aspects. Ensure each aspect-opinion pair has the appropriate sentiment polarity.
        Input: Everything tastes ok, but the place is too cramped.
        Output: food quality is ok because it is ok [SSEP] ambience general is bad because place is cramped.

        Input: {input}
        Output:""",
}


def read_line_examples_from_file(data_path):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    try:
        with open(data_path, 'r', encoding='UTF-8') as fp:
            for line in fp:
                line = line.strip()
                if line:
                    words, tuples = line.split('####')
                    sents.append(words.split())
                    labels.append(eval(tuples))
    except Exception as e:
        print(f"Error reading file {data_path}: {e}")
        return [], []
    return sents, labels


# def get_para_aste_targets(sents, labels):
#     """Get target output for ASTE task"""
#     targets = []
#     for i, label in enumerate(labels):
#         all_tri_sentences = []
#         for tri in label:
#             # a is an aspect term
#             if len(tri[0]) == 1:
#                 a = sents[i][tri[0][0]]
#             else:
#                 start_idx, end_idx = tri[0][0], tri[0][-1]
#                 a = ' '.join(sents[i][start_idx:end_idx+1])

#             # b is an opinion term
#             if len(tri[1]) == 1:
#                 b = sents[i][tri[1][0]]
#             else:
#                 start_idx, end_idx = tri[1][0], tri[1][-1]
#                 b = ' '.join(sents[i][start_idx:end_idx+1])

#             # c is the sentiment polarity
#             c = senttag2opinion[tri[2]]           # 'POS' -> 'good'

#             one_tri = f"It is {c} because {a} is {b}"
#             all_tri_sentences.append(one_tri)
#         targets.append(' [SSEP] '.join(all_tri_sentences))
#     return targets


# def get_para_tasd_targets(sents, labels):
#     """Get target output for TASD task"""
#     targets = []
#     for label in labels:
#         all_tri_sentences = []
#         for triplet in label:
#             at, ac, sp = triplet

#             man_ot = sentword2opinion[sp]   # 'positive' -> 'great'

#             if at == 'NULL':
#                 at = 'it'
#             one_tri = f"{ac} is {man_ot} because {at} is {man_ot}"
#             all_tri_sentences.append(one_tri)

#         target = ' [SSEP] '.join(all_tri_sentences)
#         targets.append(target)
#     return targets


def get_para_asqp_targets(sents, labels):
    """
    Obtain the target sentence under the paraphrase paradigm for ASQP task
    """
    targets = []
    for label in labels:
        all_quad_sentences = []
        for quad in label:
            at, ac, sp, ot = quad

            man_ot = sentword2opinion[sp]  # 'POS' -> 'good'    

            if at == 'NULL':  # for implicit aspect term
                at = 'it'

            one_quad_sentence = f"{ac} is {man_ot} because {at} is {ot}"
            all_quad_sentences.append(one_quad_sentence)

        target = ' [SSEP] '.join(all_quad_sentences)
        targets.append(target)
    return targets


def get_transformed_io(data_path, data_dir):
    """Read and transform input/output in different formats"""
    inputs, targets = [], []
    
    try:
        with open(data_path, 'r', encoding='UTF-8') as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("[Fix]"):
                    # Process correction data
                    error_target, correct_target = line.replace("[Fix]", "").split("####")
                    inputs.append(error_target.strip())
                    targets.append(correct_target.strip())
                elif "####" in line:
                    parts = line.split("####")
                    if len(parts) == 3:  # Format: original_input####stage1_output####correct_output
                        # This case is for iterative correction, skip here
                        continue
                    elif len(parts) == 2:  # Standard format: input####labels
                        words, tuples = parts
                        inputs.append(words.strip())
                        
                        try:
                            # Try to parse as standard label format
                            sents = words.split()
                            quads = eval(tuples)
                            target = get_para_asqp_targets([sents], [quads])[0]
                            targets.append(target)
                        except Exception as e:
                            # If parsing fails, use original labels directly
                            targets.append(tuples.strip())
    except Exception as e:
        print(f"Error processing file {data_path}: {e}")
        return [], []
    
    return inputs, targets


def prepare_iterative_inputs(original_inputs, first_outputs):
    """
    Prepare inputs for second-stage inference
    Format: original_sentence [SENTSEP] error_quadruples
    """
    combined_inputs = []
    for orig, first_out in zip(original_inputs, first_outputs):
        # Use [SENTSEP] to separate sentence and quadruples
        combined_text = f"{orig} [SENTSEP] {first_out}"
        combined_inputs.append(combined_text)
    return combined_inputs


# Helper function to split original sentence and quadruple parts
def split_input_by_pattern(text, aspect_categories):
    """Split input text to identify original sentence and quadruple parts"""
    # Build quadruple patterns using aspect_categories
    quad_patterns = []
    for cate in aspect_categories:
        quad_patterns.append(f"{cate} is")
    
    first_quad_pos = len(text)  # Default to end of text
    matching_pattern = None
    
    for pattern in quad_patterns:
        pos = text.find(pattern)
        if pos != -1 and pos < first_quad_pos:
            first_quad_pos = pos
            matching_pattern = pattern
    
    if matching_pattern:
        original_sentence = text[:first_quad_pos].strip()
        quads_part = text[first_quad_pos:].strip()
        return original_sentence, quads_part
    else:
        # If quadruple pattern not found, return original text and empty string
        return text, ""


class ABSADataset(Dataset):
    """Dataset class for Aspect-Based Sentiment Analysis"""
    
    def __init__(self, tokenizer, data_dir, data_type, max_len=128):
        # './data/rest16/train.txt'
        self.data_path = f'data/{data_dir}/{data_type}.txt'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir

        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def _build_examples(self):
        """Build training/evaluation samples"""
        inputs, targets = get_transformed_io(self.data_path, self.data_dir)
        self.prompt_template = DATASET_PROMPTS.get(self.data_dir, DATASET_PROMPTS['rest'])

        for i in range(len(inputs)):
            # Ensure input is a complete string, not a list of words joined by spaces
            input_text = inputs[i] if isinstance(inputs[i], str) else ' '.join(inputs[i])
            target = targets[i]

            # Use more concise formatting
            input_with_prompt = self.prompt_template.format(input=input_text)
    
            try:
                tokenized_input = self.tokenizer.batch_encode_plus(
                    [input_with_prompt], 
                    max_length=self.max_len,
                    padding="max_length",
                    truncation=True, 
                    return_tensors="pt"
                )
                tokenized_target = self.tokenizer.batch_encode_plus(
                    [target], 
                    max_length=self.max_len,
                    padding="max_length",
                    truncation=True, 
                    return_tensors="pt"
                )
        
                self.inputs.append(tokenized_input)
                self.targets.append(tokenized_target)
            except Exception as e:
                print(f"Error tokenizing example {i}: {e}")
                continue


class IterativeCorrectionDataset(Dataset):
    """
    Dataset class for second-stage correction training
    Handles data format: original_input [SENTSEP] error_quadruples #### correct_quadruples
    Where:
    - [SENTSEP] separates original input and error quadruples
    - [SSEP] separates multiple quadruples
    """
    def __init__(self, tokenizer, data_dir, data_type, max_len=128, correction_prompt=None):
        # If full path is provided, use directly; otherwise build according to original format
        if data_type.endswith('.txt') and '/' in data_type:
            self.data_path = data_type
        else:
            self.data_path = f'data/{data_dir}/{data_type}.txt'
            
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.correction_prompt = correction_prompt
        
        # Add special tokens to tokenizer if needed
        if '[SENTSEP]' not in tokenizer.get_vocab():
            special_tokens = {'additional_special_tokens': ['[SENTSEP]']}
            tokenizer.add_special_tokens(special_tokens)
            print("Added [SENTSEP] to vocabulary")

        self.inputs = []
        self.targets = []

        self._build_examples()
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        
        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()
        
        return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask}
    
    def _build_examples(self):
        """
        Build second-stage training data
        Input format: original_sentence [SENTSEP] error_quad1 [SSEP] error_quad2 ...
        Output format: correct_quad1 [SSEP] correct_quad2 ...
        """
        try:
            with open(self.data_path, 'r', encoding='UTF-8') as fp:
                for line in fp:
                    line = line.strip()
                    if not line or '####' not in line:
                        continue
                    
                    # Split input and target
                    parts = line.split('####')
                    if len(parts) != 2:
                        continue
                        
                    input_str, target_str = parts[0].strip(), parts[1].strip()
                    
                    # Check if input already contains [SENTSEP]
                    if '[SENTSEP]' not in input_str:
                        # Try to find original sentence and error quadruples
                        # Assume if [SENTSEP] exists, the part before it is the original sentence without quadruples
                        if '[SSEP]' in input_str:
                            # Might be format: "original_sentence error_quad1 [SSEP] error_quad2..."
                            # Try to find the starting position of the first quadruple
                            for aspect_cat in aspect_cate_list:
                                if f"{aspect_cat} is" in input_str:
                                    pos = input_str.find(f"{aspect_cat} is")
                                    if pos > 0:
                                        # Found first aspect category, use content before it as original sentence
                                        orig_sentence = input_str[:pos].strip()
                                        error_quads = input_str[pos:].strip()
                                        input_str = f"{orig_sentence} [SENTSEP] {error_quads}"
                                        break
                        else:
                            # If unable to determine how to split, directly add [SENTSEP]
                            if any(f"{aspect_cat} is" in input_str for aspect_cat in aspect_cate_list):
                                # Input might already be error quadruples
                                input_str = f"N/A [SENTSEP] {input_str}"
                    
                    # Add correction prompt if provided
                    if hasattr(self, 'correction_prompt') and self.correction_prompt:
                        # Check if input has [SENTSEP] separator
                        if '[SENTSEP]' in input_str:
                            # Add prompt between original sentence and error quadruples
                            sent_parts = input_str.split('[SENTSEP]', 1)
                            input_str = f"{self.correction_prompt} {sent_parts[0]} [SENTSEP]{sent_parts[1]}"
                        else:
                            # If no separator, add directly at the beginning
                            input_str = f"{self.correction_prompt} {input_str}"
                    
                    # Encode input and output
                    tokenized_input = self.tokenizer.batch_encode_plus(
                        [input_str], 
                        max_length=self.max_len,
                        padding="max_length",
                        truncation=True, 
                        return_tensors="pt"
                    )
                    
                    tokenized_target = self.tokenizer.batch_encode_plus(
                        [target_str], 
                        max_length=self.max_len,
                        padding="max_length",
                        truncation=True, 
                        return_tensors="pt"
                    )
                    
                    self.inputs.append(tokenized_input)
                    self.targets.append(tokenized_target)
        except Exception as e:
            print(f"Error building examples from {self.data_path}: {e}")
