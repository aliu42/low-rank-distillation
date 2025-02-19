#This code generates history x future matrix, computes scores for all pairs and saves the matrix

!pip install --upgrade transformers accelerate bitsandbytes
!pip install -U bitsandbytes

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os




if not torch.cuda.is_available():
    print("CUDA is not available. Using CPU.")
else:
    print("CUDA is available. Using GPU.")

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)

from transformers import BitsAndBytesConfig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # Turn on 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,     # The dtype to compute in
    bnb_4bit_use_double_quant=True,           # Optional: slightly more memory-efficient
    bnb_4bit_quant_type="nf4"                 # Optional: the quantization data type
)





model_name = "openai-community/gpt2"


model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(model)

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create a folder in the root directory
address = "/content/drive/MyDrive/llm-data/gpt2-test"
!mkdir -p "$address"

model.eval()


prompt = "One day"
encode_prompt = tokenizer(prompt, return_tensors='pt').to(device)
input_tokens = encode_prompt['input_ids']
attention_mask = encode_prompt['attention_mask']


base_length = len(input_tokens[0])

length = 16
N = 4000
trials = 1
temp = 1.0



batch_size = 40

#Sample N sequences of 2*length tokens (plus the base prompt)
#Sampled in batches of size batch_size

all_tokens = input_tokens.repeat(N,1).to(device)


#Stores all N complete sequences
new_tokens = torch.zeros((N, base_length + length * 2), dtype = int)

steps = int(N/batch_size)
for i in range(steps):
  print(i)
  curr_tokens = all_tokens[i*batch_size:(i+1)*batch_size]
  curr_mask = torch.ones_like(curr_tokens)
  with torch.no_grad():
    #with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model.generate(curr_tokens, attention_mask = curr_mask, max_new_tokens = length * 2, min_new_tokens = length * 2, do_sample=True,
                            temperature=temp, num_return_sequences=trials, output_scores = True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
    test_tokens = output.sequences
    test_scores = output.scores


    new_tokens[i*batch_size:(i+1)*batch_size] = test_tokens

all_tokens= new_tokens.to(device)
print(all_tokens.shape)

#Look at sentences for sanity check
for i in range(min(10, N)):
  print(i, tokenizer.decode(all_tokens[i]))

#Compute log Pr[f_i|h_j] for all f_i, h_j
#Computed in batches of size window x window

window = 16
steps = int(N/window)

#Will contain the log probabilities
scores_matrix = torch.zeros((N,N), dtype =  torch.bfloat16)

for ind1 in range(steps):
  print(ind1)
  for ind2 in range(steps):
    #test_batch, label_batch contains pairs of sequences h_i f_j (concatenated)
    test_batch = torch.zeros((window, window, base_length + 2 * length) , dtype = input_tokens.dtype, device = device)
    label_batch = torch.zeros((window, window, base_length + 2 * length) , dtype = input_tokens.dtype, device = device)
    for rem1 in range(window):
      for rem2 in range(window):
        i = ind1*window + rem1
        j = ind2*window + rem2
        test_ids = torch.cat((all_tokens[i][:base_length + length], all_tokens[j][base_length +  length:]), dim = 0)
        label_ids = test_ids.clone()

        test_batch[rem1][rem2] = test_ids
        label_batch[rem1][rem2] = label_ids

    test_batch = test_batch.reshape(-1, base_length + 2 *length)
    label_batch = label_batch.reshape(-1, base_length + 2 *length)
    #with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    with torch.no_grad():
      logits = model(
        input_ids=test_batch,
        attention_mask=torch.ones_like(test_batch),
        labels=label_batch,  # skip BOS token
        return_dict=True,
      ).logits

    #only compute logits for the tokens in the f part
    shift_logits = logits[:, base_length + length -1:-1, :].contiguous()
    shift_labels = label_batch[:, base_length + length:].contiguous()

    loss_per_token = F.cross_entropy(
      shift_logits.view(-1, shift_logits.size(-1)),
      shift_labels.view(-1),
      reduction='none'
    )
    loss_per_token = loss_per_token.view(window * window,  length)

    loss_per_example = loss_per_token.sum(dim=1)
    loss_per_example = loss_per_example.reshape((window,window))
    scores_matrix[ind1*window:(ind1+1)*window, ind2*window:(ind2+1)*window] = loss_per_example

scores_matrix_local = -scores_matrix.detach().cpu().float().numpy()
probs_matrix_local = np.zeros((N,N))

def log_sftmx(vec):
  max_val = np.max(vec)
  answer = np.log(np.sum(np.exp(vec - max_val))) + max_val
  return answer

#Normalize columns of prob matrix to sum to 1
for i in range(N):
  shift = log_sftmx(scores_matrix_local[:,i])
  scores_matrix_local[:, i] -= shift
  vec = scores_matrix_local[:, i]
  probs_matrix_local[:,i] = np.exp(vec)/np.sum(np.exp(vec))

#Save data
np.set_printoptions(suppress=True, precision = 6)
np.savetxt(address + "/matrix.csv", probs_matrix_local, delimiter=",", fmt='%.6f')
tokens_matrix=[]
for i in range(N):
  next_tokens = all_tokens[i].detach().cpu().numpy()
  tokens_matrix.append(next_tokens)
np.savetxt(address + "/tokens.csv", tokens_matrix, delimiter=",", fmt='%.6f')
np.savetxt(address + "/scores.csv", scores_matrix_local, delimiter=",", fmt='%.6f')

