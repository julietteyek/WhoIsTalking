""" cleans, splits & tokenizes the data """

import pandas as pd
import re
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler

# list of meaningless expressions
floskeln = [
    r"Herr Alterspräsident",
    r"Frau Präsidentin!?|Sehr geehrte(r)? (Frau|Herr)? Präsident(in)?!?|Werte Präsident(in)?!?",
    r"Meine Damen und Herren!?|Meine sehr geehrten Damen und Herren\.?",
    r"Sehr geehrte Damen und Herren",
    r"Liebe(r) Minister(in)!?",
    r"Liebe Kolleginnen und Kollegen!?|Herr Kollege!?|Frau Kollegin!?",
    r"Sehr geehrtes Präsidium!?",
    r"Sehr geehrter Herr Bundeskanzler!?",
    r"Liebe Bürgerinnen und Bürger!?|Liebe Zuschauerinnen und Zuschauer!?|Sehr geehrte Demokratinnen und Demokraten!?|Liebe Zuschauer/‑innen!?",
    r"Herr Vorsitzender\.?|Frau Vorsitzende!?",
    r"Abgeordnete!?"
]

def clean_text(text): #clean text from meaningless expressions
    for phrase in floskeln:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)
    # remove double-spaces that can occur from removing the text
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_data(input_file):
    """Cleans text:
        - removes meaningless expressions
        - compromises the parties to only the valid ones (SPD, CDU/CSU, FDP, DIE LINKE, AfD, BÜNDNIS 90/DIE GRÜNEN)
        - df mit nur test & speaker_party 
        - oversamples the parties
    """
    
    cleaned_csv_filepath = 'data/cleaned_all_data_merged.csv'
    cleaned_pkl_filepath = 'data/cleaned_all_data_merged.pkl' #nrows=500

    df = pd.read_pickle(input_file)
    print(f"nach dem EINLESEN: {df.head(5)}")

    tqdm.pandas(desc="Bereinige Text")

    df = df.dropna(subset=['text', 'speaker_party']) # = df without None in party-column
    print(f"nach dem DROPNA: {df.head(5)}")
    df['text'] = df['text'].astype(str) # set text-column as string
    print(f"nach dem ASTYPE STR: {df.head(5)}")

    df['clean_text'] = df['text'].progress_apply(clean_text) # = df without meaningless expressions
    print(f"nach dem CLEAN_TEXT: {df.head(5)}")

    def map_party(speaker_party):
        party_lower = speaker_party.lower()
        if 'bündnis' in party_lower or 'gruene' in party_lower or 'grüne' in party_lower:
            return 'BÜNDNIS 90/DIE GRÜNEN'
        elif 'linke' in party_lower:
            return 'DIE LINKE'
        elif 'cdu' in party_lower or 'csu' in party_lower:
            return 'CDU/CSU'
        elif 'spd' in party_lower:
            return 'SPD'
        elif 'fdp' in party_lower:
            return 'FDP'
        elif 'afd' in party_lower:
            return 'AfD'
        return speaker_party

    df['speaker_party'] = df['speaker_party'].apply(map_party)
    valid_parties = ['SPD', 'CDU/CSU', 'FDP', 'DIE LINKE', 'AfD', 'BÜNDNIS 90/DIE GRÜNEN']
    df = df[df['speaker_party'].isin(valid_parties)] # df = nur mit validen Parteien-Labeln
    print(f"nach dem ONLY VALID PARTIES: {df.head(5)}")

    print("VOR Oversampling:")

    #oversampling:
    print("Oversampling startet...")
    oversampler = RandomOverSampler(random_state=42)
    texts_resampled, labels_resampled = oversampler.fit_resample(
        df[['clean_text']], df['speaker_party']
    )

    df_resampled = pd.DataFrame({'text': texts_resampled['clean_text'], 'speaker_party': labels_resampled})

    print("NACH Oversampling:")
    print(df_resampled['speaker_party'].value_counts())

    # save new cleaned df to .pkl and .csv
    df_resampled.to_csv(cleaned_csv_filepath, index=False)
    print(f"Bereinigte CSV gespeichert unter: {cleaned_csv_filepath}")
    df_resampled.to_pickle(cleaned_pkl_filepath)
    print(f"Bereinigte Pickle gespeichert unter: {cleaned_pkl_filepath}")

    return df_resampled