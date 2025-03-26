# project.py


import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    def check_robots():
        global delay_time
        robots_url = "https://gutenberg.org/robots.txt"
        response = requests.get(robots_url)
        if response.status_code == 200:
            match = re.search(r"Crawl-delay: (\d+)", response.text)
            if match:
                delay_time = int(match.group(1))
            else:
                delay_time = 0.5
# Check robots.txt for Crawl-delay before making requests
    check_robots()
    time.sleep(delay_time)# Pause before making the request
    response = requests.get(url)
    if response.status_code != 200:
        time.sleep(delay_time)
        response = requests.get(url)
    text = response.text
    # Normalize line endings
    text = text.replace('\r\n', '\n')

# Extract the content between the START and END markers
    start_indx = text.find("***\n") + 3
    end_indx = text.find("*** END OF THE PROJECT GUTENBERG")
    return text[start_indx:end_indx] 


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    if not book_string or book_string.isspace():
        return ['\x02', '\x03']
    
    text_with_markers = re.sub(r'\n\s*\n+', '\x03\x02', book_string.strip())

    marked_text = '\x02' + text_with_markers + '\x03'
    pattern = r'(\x02|\x03|\w+(?:_\w+)*|[^\w\s_]|\w+)'
    return re.findall(pattern, marked_text)


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):
    def __init__(self, tokens):
        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        # uniq_tokens = set(tokens)
        # num_uniq = len(uniq_tokens)
        # if num_uniq != 0:
        #     probab = np.ones(num_uniq) * (1/num_uniq)
        # else:
        #     probab = np.array()
        
        # return pd.Series(probab, index=uniq_tokens)
        token = set(tokens)
        return pd.Series([1 / len(token)] * len(token), index=token)
    
        
    def probability(self, words):
        prob_seq = 1
        for token in words:
            if token not in self.mdl.index:
                return 0
                # prob_token = self.mdl.loc[token]
            # else:
                prob_token = 0
            prob_seq *= self.mdl.loc[token]

        return prob_seq

    def sample(self, M):
        return " ".join(np.random.choice(list(self.mdl.index), M, list(self.mdl)))

# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------

class UnigramLM(object):
    def __init__(self, tokens):
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
       return pd.Series(tokens).value_counts(normalize=True)
    
    def probability(self, words):
        prob_seq = 1
        for token in words:
            if token in self.mdl.index:
                prob_token = self.mdl.loc[token]
            else:
                return 0
            prob_seq *= prob_token
        return prob_seq

    def sample(self, M):
        return ' '.join(self.mdl.sample(n=M, replace=True, weights=self.mdl).index)

# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        all_tuples = []
        for i in range(len(tokens) - (self.N-1)):
            the_lst_tuple = []
            for j in range(self.N):
                the_lst_tuple.append(tokens[i+j])
            
            the_tuple = tuple(the_lst_tuple)
            all_tuples.append(the_tuple)
        
        return all_tuples

        
    def train(self, ngrams):
        # N-Gram counts C(w_1, ..., w_n)
        cleaned_ngrams = [tuple('' if token in ['\x02', '\x03'] else token for token in ngram) for ngram in ngrams]
        df_ngrams = pd.DataFrame()
        df_ngrams['ngram'] = pd.Series(ngrams)
        value_count = df_ngrams['ngram'].value_counts()
        df_ngrams['ngram_count'] = df_ngrams['ngram'].map(value_count)

        # (N-1)-Gram counts C(w_1, ..., w_(n-1))
        n_min1 = []
        for ngram in list(df_ngrams['ngram']):
            new_tuple = ngram[:-1]
            n_min1.append(new_tuple)

        df_ngrams['n1gram'] = pd.Series(n_min1)
        value_count_min1 = df_ngrams['n1gram'].value_counts()
        df_ngrams['n1gram_count'] = df_ngrams['n1gram'].map(value_count_min1)

        # Create the conditional probabilities
        ngram_probs = df_ngrams['ngram_count']/df_ngrams['n1gram_count']
        df_ngrams['prob'] = ngram_probs
        
        # Put it all together
        return df_ngrams[['ngram', 'n1gram', 'prob']].drop_duplicates(subset=['ngram'])
    
    def probability(self, words):
        prob = 1
        num = self.N
        token_lst = []
        for i in range(len(words)):
            token_lst.append(tuple(words[max(0, i-num+1):i+1]))
        
        for token in token_lst:
            count = num
            current = self
            if len(token) == 1:
                while count > 1:
                    current = current.prev_mdl
                    count -= 1

                prob *= current.mdl.get(token[0],0)
            
            else:
                morethan1 = count - len(token)
                for j in range(morethan1):
                    current = current.prev_mdl
                df = current.mdl
                need = df[(df['ngram'] == token) & (df['n1gram'] == token[:-1])]
                try:
                    prob *= need.iloc[0, -1]
                except IndexError:
                    return 0
            
        return prob
    

    def sample(self, M):

        # Use a helper function to ge/sample tokens of length `length`
        all_tokens = ['\x02']
    
        for i in range(M-1):
            current = self
            # num_iter = self.N - 3
            # for j in range(num_iter):
            #     current = current.prev_mdl


            # if there's already one value in the list
            if len(all_tokens) == 1:
                for j in range(self.N - 2):
                    current = current.prev_mdl

                df = current.mdl
                # display(df)
                needed = df[df['ngram'].str[0] == '\x02']
                needed_vals = needed['ngram'].str[1].value_counts()
                # display(needed_vals)
                cond_prob = needed_vals/needed_vals.sum()
            
            #if there's less than N values in the list
            elif len(all_tokens) < self.N:

                for j in range(self.N - len(all_tokens) - 1):
                    current = current.prev_mdl

                df = current.mdl
                # display(df)
                # print(i)
                needed = df[df['n1gram'].str[:current.N-1] == tuple(all_tokens[-(current.N - 1):])]
            
                #display(needed)
                needed_vals = needed['ngram'].str[-1].value_counts()
                cond_prob = needed_vals/needed_vals.sum()
            else:
                
                df = current.mdl
                # display(df)
                # print(all_tokens[-(self.N - 1):])
                # display(df['n1gram'].str[:self.N-1])
                # display(all_tokens[-(self.N - 1):])
                needed = df[df['n1gram'].str[:self.N-1] == tuple(all_tokens[-(self.N - 1):])]
                # display(needed)

                needed_vals = needed['ngram'].str[-1].value_counts()
                cond_prob = needed_vals/needed_vals.sum()
                # for k in range(self.N):
                #     display(needed)
                #     print(all_tokens[-(self.N-k)])
                #     
                # display(needed)
            if len(needed) == 0:
                all_tokens.append('\x03')
            
            else:
                generate_token = np.random.choice(cond_prob.index, p=cond_prob.values)
                # print(generate_token)
                all_tokens.append(generate_token)
            # print(all_tokens)
        # Transform the tokens to strings
        all_tokens.append('\x03')
        
        return ' '.join(all_tokens)


            
        #     if len(all_tokens) == 1:
        #         needed = df[df['ngram'].str[0] == '\x02']
        #         # display(needed)
        #         needed_vals = needed['ngram'].str[1].value_counts()
        #         cond_prob = needed_vals/needed_vals.sum()
            
        #     else:
        #         # print(all_tokens)
        #         needed = df[df['ngram'].str[0] == all_tokens[-(self.N - 1)]]
        #         for k in range(1, min(self.N-1, i-1)):
        #             needed = needed[needed['ngram'].str[k] == all_tokens[-(self.N + k)]]
                
        #         needed_vals = needed['ngram'].str[-1].value_counts()
        #         cond_prob = needed_vals/needed_vals.sum()

        #     if len(needed) == 0:
        #         all_tokens.append('\x03')
            
        #     else:
        #         generate_token = np.random.choice(cond_prob.index, p=cond_prob.values)
        #         all_tokens.append(generate_token)
        
        # # Transform the tokens to strings
        # if all_tokens[-1] != '\x03':
        #     all_tokens.append('\x03')
        # return ' '.join(all_tokens)






        # def find_next(self, context):
        #     model = self
        #     current_count = self.N - 1
        #     if len(context) < self.N:
        #         while current_count > len(context):
        #             model = model.prev_mdl
        #             current_count -= 1

        #     model = model.mdl
        #     model = model[model['n1gram'] == context]
        #     choices = [ngram[-1] for ngram in model['ngram']]
        #     choices_prob = model['prob'].tolist()
        #     if len(choices) == 0:
        #         return '\x03'
        #     return str(np.random.choice(choices, p=choices_prob))

        # tokens = ['\x02']
        # token_count = 0
        # while token_count < M - 1:
        #     tokens.append(find_next(self, tuple(tokens[-(self.N + 1):])))
        #     token_count += 1
        #     if (token_count < M - 2) & (tokens[-1] == '\x03'):
        #         tokens.append('\x02')
        #         token_count += 1
        # tokens.append('\x03')
        # return ' '.join(tokens)
