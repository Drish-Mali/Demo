import nltk
from nltk import FreqDist
from nltk.util import ngrams
nltk.download('punkt')
import nltk
nltk.download('punkt')

# Open the text file in read mode
file_path = "corpus_for_language_models.txt"


lines_as_lists = []

try:
    with open(file_path, 'r') as file:
        for line in file:
            # Tokenize the line using word_punct_tokenize
            tokens =  nltk.word_tokenize(line)
            # Append the list of tokens to the lines_as_lists
            lines_as_lists.append(tokens)
except FileNotFoundError:
    print(f"The file '{file_path}' does not exist.")
except IOError as e:
    print(f"An error occurred while reading the file: {e}")

# Print the list of lists
for line_list in lines_as_lists:
    print(line_list)


def get_output(input_list):
  """Converts a list of lists of strings to a list of strings, with each string containing all the elements from the corresponding lists in the input list, separated by a space.

  Args:
    input_list: A list of lists of strings.

  Returns:
    A list of strings, with each string containing all the elements from the corresponding lists in the input list, separated by a space.
  """

  output_list = []
  for sublist in input_list:
    output_string = " ".join(sublist)
    output_list.append(output_string)
  return output_list

out_put_list=get_output(lines_as_lists)
out_put_list
def find_vocab(input_list):
  """Finds all the unique vocab in a list of lists of strings.

  Args:
    input_list: A list of lists of strings.

  Returns:
    A set of all the unique vocab in the input list.
  """

  vocab_set = set()
  for sublist in input_list:
    for word in sublist:
      vocab_set.add(word)
  return vocab_set
vocab_set = find_vocab(lines_as_lists)
print(len(vocab_set))
print(vocab_set)
import pandas as pd
from tqdm import tqdm

def co_occurance_matrix(input_text, top_words, window_size):
    co_occur = pd.DataFrame(index=top_words, columns=top_words)

    for row, nrow in tqdm(zip(top_words, range(len(top_words))), total=len(top_words), desc="Building Matrix"):
        for colm, ncolm in zip(top_words, range(len(top_words))):
            count = 0
            if row == colm:
                co_occur.iloc[nrow, ncolm] = count
            else:
                for single_essay in input_text:
                    essay_split = single_essay.split(" ")
                    max_len = len(essay_split)
                    top_word_index = [index for index, split in enumerate(essay_split) if row in split]
                    for index in top_word_index:
                        if index == 0:
                            count = count + essay_split[:window_size + 1].count(colm)
                        elif index == (max_len - 1):
                            count = count + essay_split[-(window_size + 1):].count(colm)
                        else:
                            count = count + essay_split[index + 1: (index + window_size + 1)].count(colm)
                            if index < window_size:
                                count = count + essay_split[:index].count(colm)
                            else:
                                count = count + essay_split[(index - window_size): index].count(colm)
                co_occur.iloc[nrow, ncolm] = count

    return co_occur


window_size =5

result = co_occurance_matrix(out_put_list,list(vocab_set),window_size)
result.to_csv("co_oocurance.csv",index=False)
import numpy as np
import math
def ppmi(word,context_word,df):
  num_prob=df[word,context_word]/df.sum(axis='both')
  denum_prob=(df[word,:]/df.sum(axis='both'))*(df[context_word,:]/df.sum(axis='both'))
  pmi = np.log2(num_prob/denum_prob)
  ppmi = np.maximum(pmi, 0)

  return ppmi
print(round(ppmi('sales','president',result),2))
print(round(ppmi('sales','said',result),2))
print(round(ppmi('company','president',result),2))
print(round(ppmi('company','of',result),2))
def cosine_similairity_with_context(context_list,word1,word2,df):
  vector_word1=[]
  vector_word2=[]
  for context in context_list:
    vec1=ppmi(word1,context,df)
    vec2=ppmi(word2,context,df)
    vector_word1.append(vec1)
    vector_word2.append(vec2)
  # Convert the lists to NumPy arrays
  v1_array = np.array(vector_word1)
  v2_array = np.array(vector_word2)

  # Calculate the dot product of the two arrays
  dot_product = np.dot(v1_array, v2_array)

  # Calculate the magnitudes of the two arrays
  magnitude_v1 = np.linalg.norm(v1_array)
  magnitude_v2 = np.linalg.norm(v2_array)

  # Calculate the cosine similarity
  cosine_similarity = dot_product / (magnitude_v1 * magnitude_v2)
  return cosine_similarity
context_list=['said','of','president']
print(round(cosine_similairity_with_context(context_list,'executive','company',result),2))
print(round(cosine_similairity_with_context(context_list,'sales','purchase',result),2))
print(round(cosine_similairity_with_context(context_list,'executive','sales',result),2))