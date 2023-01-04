from params import *
import string
from transformers import CamembertTokenizer
from torch.utils.data import TensorDataset

stopwords_list = ['au', 'aux', 'avec', 'ce', 'ces', 'dans', 'de', 'des', 'du', 'elle', 'en', 'et', 'eux', 'il', 'ils', 'je', 'la', 'le', 'les', 'leur', 'lui', 'ma', 'mais', 'me', 'même', 'mes', 'moi', 'mon', 'nos', 'notre', 'nous', 'on', 'ou', 'par', 'pour', 'qu', 'que', 'qui', 'sa', 'se', 'ses', 'son', 'sur', 'ta', 'te', 'tes', 'toi', 'ton', 'tu', 'un', 'une', 'vos', 'votre', 'vous', 'c', 'd', 'j', 'l', 'à', 'm', 'n', 's', 't', 'y', 'été', 'étée', 'étées', 'étés', 'étant', 'étante',
'étants', 'étantes', 'suis', 'es', 'est', 'sommes', 'êtes', 'sont', 'serai', 'seras', 'sera', 'serons', 'serez', 'seront', 'serais', 'serait', 'serions', 'seriez', 'seraient', 'étais', 'était',
'étions', 'étiez', 'étaient', 'fus', 'fut', 'fûmes', 'fûtes', 'furent', 'sois', 'soit', 'soyons', 'soyez', 'soient', 'fusse', 'fusses', 'fût', 'fussions', 'fussiez', 'fussent', 'ayant', 'ayante', 'ayantes', 'ayants', 'eu', 'eue', 'eues', 'eus', 'ai', 'as', 'avons', 'avez', 'ont', 'aurai', 'auras', 'aura', 'aurons', 'aurez', 'auront', 'aurais', 'aurait', 'aurions', 'auriez', 'auraient', 'avais', 'avait', 'avions', 'aviez', 'avaient', 'eut', 'eûmes', 'eûtes', 'eurent', 'aie', 'aies', 'ait', 'ayons', 'ayez', 'aient', 'eusse', 'eusses', 'eût', 'eussions', 'eussiez', 'eussent']
#regex_exp = re.compile(r'|'.join(stopwords_list),re.IGNORECASE)
#regex_exp = re.compile("|".join(stopwords_list))

def stopwords_remove(x):
    x.translate(x.maketrans('','',string.punctuation))
    splits = x.split()
    cleaned = ["" if x.lower() in stopwords_list else x for x in splits]
    cleaned = [x for x in cleaned if x != '']
    cleaned = ' '.join(cleaned)
    return cleaned

def remove_stopwords(df):
    df["commentaire"] = df["commentaire"].apply(stopwords_remove)
    return df

def dataset_to_pickle_2(dataset_name, note=True):
    if not os.path.exists(os.path.join(pickle_folder, dataset_name+"_2.p")):
        logging.info(f"Loading xml of: {dataset_name}")
        df_data = pd.read_xml(os.path.join(xml_folder, dataset_name + ".xml"))
        df_data.fillna('a', inplace=True)
        os.makedirs(pickle_folder, exist_ok=True)
        logging.info(f"Removing stopwords in: {dataset_name}")
        df_data = remove_stopwords(df_data)
        logging.info(f"Tokenization of: {dataset_name}")
        reviews = df_data['commentaire'].values.tolist()
        tokenizer = CamembertTokenizer.from_pretrained(
            'camembert-base',
            do_lower_case=True)
        encoded_batch = tokenizer.batch_encode_plus(reviews,
                                                    add_special_tokens=True,
                                                    max_length=512,
                                                    padding=True,
                                                    truncation=True,
                                                    return_attention_mask=True,
                                                    return_tensors='pt')
        if note:
            df_token = pd.DataFrame(columns=['commentaire', 'note'])
            df_token['note'] = df_data['note'].apply(to_float_2)
            sentiments = df_token['note'].values.tolist()
            sentiments = torch.tensor(sentiments)
            dataset = TensorDataset(
                encoded_batch['input_ids'],
                encoded_batch['attention_mask'],
                sentiments)
        else:
            dataset = TensorDataset(
                encoded_batch['input_ids'],
                encoded_batch['attention_mask'])
        with open(os.path.join(pickle_folder, dataset_name+"_2.p"), 'wb') as f:
            pickle.dump(dataset, f)
        logging.info(f"Saving pickle: {dataset_name}")
    else:
        logging.info(f"Loading pickle: {dataset_name}")
        with open(os.path.join(pickle_folder, dataset_name+"_2.p"), 'rb') as f:
            dataset = pickle.load(f)
    # df_token['commentaire'] = df_data['commentaire']
    # print(df_token)
    return dataset


if __name__ == '__main__':
    # train_dataset = dataset_to_pickle_2("train")
    valid_dataset = dataset_to_pickle_2("dev")