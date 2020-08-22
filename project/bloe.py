from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd

class BLEU():
    def __init__(self, gold_captions_file, output_captions_file, weight=(.25, .25, .25, .25)):
        self.gold_captions_file = gold_captions_file
        self.output_captions_file = output_captions_file
        self.weight = weight 

        with open(self.output_captions_file, 'r') as f1, open(self.gold_captions_file, 'r') as f2:
            self.df_output = pd.read_csv(f1, sep='\t')
            self.df_gold = pd.read_csv(f2, sep='\t')

    
    def make_dict(self, df):
        d = {}
        try:
            for image in df['image']:
                if image not in d.keys():
                    d[image] = [word_tokenize(i) for i in list(df[df['image'] == image]['caption'])]
        except TypeError:
            pass

        return d


    def make_bleu_dict(self):
        '''
        return dictionary with image id as key and bleu score as value
         {'000000203564.jpg': 0.5410822690539396}
        
        '''
        _bleu_dict = {}
        gold_dict = self.make_dict(self.df_gold)
        output_dict = self.make_dict(self.df_output)
        
        for _image in self.df_output['image']:
            reference = gold_dict[_image]
            candidate = output_dict[_image][0]
            
            _bleu_dict[_image] = sentence_bleu(reference, candidate, weights=self.weight)
            
        return _bleu_dict

    
    def get_bleu_score(self):
        '''

        returns average of BLEU-4 score over all output captions
        
        '''
        bleu_dict = self.make_bleu_dict()
        scores = bleu_dict.values()

        return sum(scores) / len(scores)


# blö = BLEU('captions.txt', 'tiny_output_captions.txt')

# score = blö.get_bleu_score()
# print(score)

# #0.6458