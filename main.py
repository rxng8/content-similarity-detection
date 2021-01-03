# %%
import tensorflow as tf
from transformers import AutoModel, AutoTokenizer
from transformers import TFAutoModel

phobert = TFAutoModel.from_pretrained("vinai/phobert-base", output_attentions=False)
# phobert = AutoModel.from_pretrained("vinai/phobert-base")

# For transformers v4.x+: 
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

# %%


# INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
line = "Tôi là sinh_viên trường đại_học Công_nghệ ."

input_ids = tf.convert_to_tensor([tokenizer.encode(line)])

# %%
# You can’t unpack a ModelOutput directly. Use the to_tuple() method to convert it to a tuple before.
"""
last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) 
    – Sequence of hidden-states at the output of the last layer of the model.

pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) – Last layer hidden-state 
    of the first token of the sequence (classification token) further processed by a Linear layer 
    and a Tanh activation function. The Linear layer weights are trained from the next sentence 
    prediction (classification) objective during pretraining.
"""
last_hidden_state, pooler_output = phobert(input_ids).to_tuple()


# %%

last_hidden_state
