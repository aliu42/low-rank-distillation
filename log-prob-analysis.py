#Here we analyze the history x future matrix of scores

import numpy as np

from google.colab import drive
drive.mount('/content/drive')

address = "/content/drive/MyDrive/llm-data/gpt2-test"


#scores and probs are NxN matrices containing log Pr[f|h] and Pr[f|h] respectively
#tokens matrix is N x total length, with each row storing h_if_i (concatenated)
scores_matrix = np.loadtxt(address + "/scores.csv", delimiter=",")
tokens_matrix = np.loadtxt(address + "/tokens.csv", delimiter=",")
probs_matrix = np.loadtxt(address + "/matrix.csv", delimiter=",")
tokens_matrix = tokens_matrix.astype(int)


print(np.shape(scores_matrix))
print(np.shape(tokens_matrix))
print(np.shape(probs_matrix))

N = np.shape(scores_matrix)[0]

U, sig, V = np.linalg.svd(scores_matrix)

import matplotlib.pyplot as plt


max_rank = 500
#truncate removes the top few singular values
truncate = 20

#Correct normalization is to divide by N
vals = sig[truncate:max_rank]/N
x_vals = np.array([i for i in range(truncate, max_rank)])

#Fit power law
slope, intercept = np.polyfit(np.log(x_vals), np.log(vals), 1)

print("slope" , slope)
print("intercept", intercept)

plt.plot(x_vals,vals)
#plt.plot(np.log(x_vals), np.log(vals))
#plt.plot(np.log(x_vals), slope * np.log(x_vals) + intercept , label = "Fit Line")
plt.xlabel('Rank')
plt.ylabel('Singular Value')
plt.legend()
plt.show()

from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)



#These are set manually right now
#base_length = length of prompt, length = length of each history or future
base_length = 2
length = 16

#Low rank approx to scores matrix
rank = 500
approx_scores = np.matmul(np.matmul(U[:,:rank], np.diag(sig[:rank])), V[:rank, :])


def log_sftmx(vec):
  max_val = np.max(vec)
  answer = np.log(np.sum(np.exp(vec - max_val))) + max_val
  return answer

#Computes estimated KL for a fixed history between true posterior and low rank approx
def compute_KL(ind):
  prob_vec = probs_matrix[ind]
  true_scores = scores_matrix[ind] - log_sftmx(scores_matrix[ind])
  pred_scores = approx_scores[ind] - log_sftmx(approx_scores[ind])
  return np.dot(prob_vec, true_scores - pred_scores)

#Samples futures conditioned on a fixed history according to low rank approx
#ind specifies the index of the history
def sample(ind, num_trials):
  scores_vec = approx_scores[ind]
  prob_vec = np.exp(scores_vec - log_sftmx(scores_vec))
  examples = np.random.choice(N, num_trials, p = prob_vec/np.sum(prob_vec))
  for i in range(num_trials):
    print(ind, examples[i])
    print(tokenizer.decode(tokens_matrix[ind][:base_length + length]) + tokenizer.decode(tokens_matrix[examples[i]][base_length + length:]))

for i in range(N):
  print(i, compute_KL(i))

#Print out samples drawn from low rank approx
test_ind = 0
sample(test_ind,10)
print(compute_KL(test_ind))
print(tokenizer.decode(tokens_matrix[test_ind]))

