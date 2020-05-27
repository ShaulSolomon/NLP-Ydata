'''Intializations'''
import numpy as np
from nltk import ngrams


''' HELPER FUNCTIONS '''

def nested_set(dic, keys, value = 1, create_missing=True):
    '''
    For a nested dictionary, checks if there is a nested dictionary and if not creates one.
    At the end it ultimately does += 1 is already exists or = 1 if its the first occurence.
    '''
    d = dic
    for key in keys[:-1]:
        if key in d:
            d = d[key]
        elif create_missing:
            d = d.setdefault(key, {})
        else:
            return dic
    if keys[-1] in d:
        d[keys[-1]] += value
    elif create_missing:
        d[keys[-1]] = 1
    return dic

def set_freq(dic, keys, value = 1):
    '''
    Sets the value within a nested loop
    '''
    d = dic
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value
    return dic

def get_freq(dic,keys):
    '''
    Gets the freq for a given pair
    '''
    d = dic
    for key in keys[:-1]:
        d = d[key]
    return d[keys[-1]]

def norm_nested_counts(main_dic,c_dic,N,n_keys = []):
    '''
    Iterates through the dictionary until last layer and calculates frequency with key,value pair
    '''
    for key, values in c_dic.items():
        #recurse until last layer
        if N > 2:
            n_keys.extend([key])
            norm_nested_counts(main_dic,values,N-1,n_keys)
            n_keys.pop()
        else:
            total_occ = sum([val for val in values.values()])
            for k,v in values.items():
                freq = v / total_occ
                set_freq(main_dic,n_keys + [key] + [k],freq)

def get_freq2(dic,keys):
    '''
    Gets the freq for a given pair
    '''
    d = dic
    keys = keys.flatten()
    for key in keys[:-1]:
        d = d[key]
    return d[keys[-1]].items()

''' class '''

class HMM_MODEL():

    def __init__(self,N=2):
        self.N = N
        self.model = None
        np.random.seed(26) # 26 = Havaya

    
    def create_model(self, sentences):
        #Before we put everything into pairs, we need to add a START and END to every sentence
        #sentences = [["START"] + sentence + ["END"] for sentence in sentences]
        #included them as padding
        sentences_pairs = [ngrams(sentence, self.N, pad_left=True,pad_right=True,left_pad_symbol="START", \
                                            right_pad_symbol="END") for sentence in sentences]
        
        #Dist_dict will hold all the possibilities of every N pairs option.
        dist_dict = dict()

        #Count occurence
        for sp in sentences_pairs:
            for word_pairs in sp:
                nested_set(dist_dict,word_pairs)

        #Need to normalize the counts so that the count is represented as the frequency of it's appearance
        norm_nested_counts(dist_dict,dist_dict,self.N)

        self.model = dist_dict

    def generate(self,MAX_SENTENCES = 4):

        n1_words = np.array([[val] for val in np.repeat(['START'],self.N-1)])

        model_text = []
        model_text.append([])

        sentence_count = 0

        while True:

            if n1_words[-1] == "END":
                sentence_count += 1
                n1_words =  np.array([[val] for val in np.repeat(['START'],self.N-1)])
                if sentence_count == MAX_SENTENCES:
                    break
                model_text.append([])
            elif n1_words[-1] != "START":
                model_text[sentence_count].append(n1_words[-1][0])
        
            possible_words = get_freq2(self.model,n1_words)
            guesses = np.array([value[0] for value in possible_words])
            preds = np.array([value[1] for value in possible_words])

            #Can only choose from the 32 most likely options - WEIRD
            best_idx = np.argsort(preds)[-32:]
            norm_preds = preds[best_idx] / sum(preds[best_idx])
            new_word = np.random.choice(guesses[best_idx],p = norm_preds)
            
            n1_words = np.concatenate((n1_words, np.array([new_word]).reshape(1,-1)))
            n1_words = n1_words[1:]

        para = [" ".join(text) for text in model_text]

        return para