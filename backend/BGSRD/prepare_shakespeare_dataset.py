import pandas as pd
import re

fakes = [
    'The Merry Devill of Edmonton by Shakespeare', 
    'Fair Em by Shakespeare', 
    'The Two Noble Kinsmen by Shakespeare',
    'Locrine Mucedorus by Shakespeare',
    'The Puritaine Widdow by Shakespeare',
    'Sir John Oldcastle by Shakespeare',
    'Sir Thomas More by Shakespeare',
    'A Yorkshire Tragedy by Shakespeare' 
    ]
reals = [
    'THE TWO GENTLEMEN OF VERONA',
    'THE TRAGEDY OF ROMEO AND JULIET',
    'THE TRAGEDY OF JULIUS CAESAR',
    'THE TRAGEDY OF MACBETH',
    'THE TRAGEDY OF OTHELLO MOOR OF VENICE',
    'THE TRAGEDY OF KING LEAR'
]

def clean_str(string):
    string = re.sub(r'[\t\n]', ' ', string)
    string = re.sub(r'\s+', ' ', string)
    return string.strip()


def preprocess(books, label):
    all_df = []
    chunk_size = 128
    for book in books:
        chunks = []
        with open('./{}.txt'.format(book), mode = 'r') as f:
            while chunk := f.read(chunk_size):
                chunks.append(clean_str(chunk))
        df_real = pd.DataFrame(chunks, columns = ['text'])
        df_real['title'] = book
        all_df.append(df_real)
    all_df = pd.concat(all_df, axis = 0)
    all_df['label'] = label
    return all_df

real_df = preprocess(reals, 0)
fake_df = preprocess(fakes, 1)
df = pd.concat([real_df, fake_df], axis = 0)
df['type'] = 'train'
df = df.sample(frac = 1).reset_index(drop = True)
df.to_csv('./data/train.csv', index = False)

