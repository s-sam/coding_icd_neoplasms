import numpy as np
from nltk import download, pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


def lemmatise_using_pos(pos_tagged):
    wordnetlemmatiser = WordNetLemmatizer()
    lemmatised_text = []
    wordnet_pos_mapping = {
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "J": wordnet.ADJ,
        "R": wordnet.ADV
    }
    for word, pos in pos_tagged:
        pos_key = pos[0:1]
        pos = wordnet_pos_mapping.get(pos_key, wordnet.NOUN)
        lemmatised_word = str(wordnetlemmatiser.lemmatize(word, pos=pos))
        lemmatised_text.append(lemmatised_word)             
    return ' '.join(lemmatised_text)


class CleanData():
    def __init__(self, study_data, fillna_value='', concat_separator=' ', remove_empty_rows=True,
                 lowercase=True, symbol_replace=True, symbol_fill_value='',
                 whole_number_replace=True, whole_number_fill_value='',
                 multiple_spaces_to_1=True, remove_start_end_spaces=True,
                 remove_stopwords=True, run_lemmatiser=True):
        
        self.study_data = study_data
        self.fillna_value = fillna_value
        self.concat_separator = concat_separator
        self.remove_empty_rows = remove_empty_rows
        self.lowercase = lowercase
        self.symbol_replace = symbol_replace
        self.symbol_fill_value = symbol_fill_value
        self.whole_number_replace = whole_number_replace
        self.whole_number_fill_value = whole_number_fill_value
        self.multiple_spaces_to_1 = multiple_spaces_to_1
        self.remove_start_end_spaces = remove_start_end_spaces
        self.remove_stopwords = remove_stopwords
        self.run_lemmatiser = run_lemmatiser
        self.text_cols = ['short_name', 'title', 'research_summary', 'inclusion_criteria']
    
    def keep_text_columns_only(self):
        self.study_data = self.study_data.loc[:,self.text_cols]
        return self

    def concatenate_text_fields(self):
        self.study_data['text'] = (self.study_data[self.text_cols]
                        .fillna(self.fillna_value)
                        .agg(self.concat_separator.join, axis=1)
                        )
        return self
        
    def drop_text_columns(self):
        self.study_data = self.study_data.drop(columns=self.text_cols)
        return self

    def remove_rows_with_empty_text(self):
        empty_text_represenation = self.concat_separator.join([self.fillna_value] * 4)
        self.study_data = self.study_data.replace(empty_text_represenation, np.NaN, regex=False)
        self.study_data = self.study_data.dropna()
        return self

    def convert_to_lowercase(self):
        self.study_data.text = self.study_data.text.str.lower()
        return self

    def replace_symbols(self):
        self.study_data.text = self.study_data.text.str.replace(r'[^a-zA-Z0-9 ]', self.symbol_fill_value, regex=True)
        return self

    def replace_whole_numbers_with_no_letters_attached(self):
        self.study_data.text = self.study_data.text.str.replace(r'\b[0-9]+\b', self.whole_number_fill_value, regex=True)
        return self

    def replace_multiple_spaces_with_1(self):
        self.study_data.text = self.study_data.text.str.replace(r'\s\s+', ' ', regex=True)
        return self

    def remove_start_and_end_spaces(self):
        self.study_data.text = self.study_data.text.str.replace(r'(^\s|\s$)', '', regex=True)
        return self
    
    def remove_nltk_stopwords(self):
        download('stopwords')
        stopword_list = stopwords.words('english')
        self.study_data.text = (self.study_data.text
                                .apply(lambda x: ' '.join([word for word in x.split() if word not in (stopword_list)]))
                            )
        return self

    def nltk_lemmatiser(self):
        download('punkt')
        download('averaged_perceptron_tagger')
        download('wordnet')
        self.study_data['pos_tagged'] = (self.study_data.text
                                                     .apply(word_tokenize)
                                                     .apply(pos_tag))
        
        self.study_data['lemmatised_text'] = self.study_data['pos_tagged'].apply(lambda x: lemmatise_using_pos(x))
        self.study_data['text'] = self.study_data.loc[:,'lemmatised_text']
        self.study_data.drop(columns=['pos_tagged', 'lemmatised_text'], inplace=True)
        return self
        # test="randomised-84 and i"
        # pos_tag(word_tokenize(test))
        # wordnetlemmatiser.lemmatize(test, wordnet_pos_mapping.get('JJ', wordnet.NOUN))

    def run(self):
        self.keep_text_columns_only()
        self.concatenate_text_fields()
        self.drop_text_columns()

        if self.remove_empty_rows == True:
            self.remove_rows_with_empty_text()

        if self.lowercase == True:
            self.convert_to_lowercase()
            
        if self.run_lemmatiser == True:
            self.nltk_lemmatiser()
            
        if self.remove_stopwords == True:
            self.remove_nltk_stopwords()

        if self.symbol_replace == True:
            self.replace_symbols()

        if self.whole_number_replace == True:
            self.replace_whole_numbers_with_no_letters_attached()

        if self.multiple_spaces_to_1 == True:
            self.replace_multiple_spaces_with_1()

        if self.remove_start_end_spaces == True:
            self.remove_start_and_end_spaces()
            
        if self.remove_empty_rows == True:
            self.remove_rows_with_empty_text()

        return self.study_data.squeeze()
    

if __name__ == "__main__":
    import load_data
    root_data = 'data/'

    study, icd_labels, icd_reviewed = load_data.load_tables(root=root_data,
                                           tables=['study', 'icd_labels', 'icd_reviewed'])
    
    text_prepared = CleanData(study_data=study).run()
    
    
    print(text_prepared.head())