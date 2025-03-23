# Load textures from freesound

FREESOUND_API_KEY = '7lUCaWwUMv1ItYZrDkGdsLhWWucoaFV2mPnVQDsQ'  # Please replace by your own Freesound API key
OGG_PATH = 'oggs'  # Place where to store the downloaded diles. Will be relative to the current folder.
WAV_PATH = "wavs"
DAC_PATH = "testdata"
DATAFRAME_FILENAME = 'testdata/dac-train.xlsx'  # File where we'll store the metadata of our sounds collection
FREESOUND_STORE_METADATA_FIELDS = ['id', 'name', 'username', 'previews', 'license', 'tags']  # Freesound metadata properties to store
FREESOUND_QUERIES = [
    {
        'query': 'helicopter',
        'filter': "duration:[6 TO 30]",
        'num_results': 50,
    },
{
        'query': 'radio',
        'filter': "duration:[6 TO 30]",
        'num_results': 50,
    },
{
        'query': 'rain',
        'filter': "duration:[6 TO 30]",
        'num_results': 50,
    },
{
        'query': 'supersaw',
        'filter': "duration:[6 TO 30]",
        'num_results': 50,
    },
]
import os
import pandas as pd
import numpy as np
import freesound
from tqdm import tqdm
import shutil
import essentia.standard as es
from sklearn.model_selection import train_test_split
import torch
from dac.utils import load_model
from audiotools import AudioSignal
import dac


def encode_audio_files(df, output_dir, model_sr='44khz', model_tag='latest', device="cuda", n_quantizers=4):
    with torch.no_grad():
        # Load the DAC model
        model_path = dac.utils.download(model_type=model_sr)
        model = dac.DAC.load(model_path).to(device)


        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Iterate over all WAV files
        for file_path in tqdm(df.ID, desc="Encoding audio files"):

            wav_path = df[df.ID == file_path]["WAV Path"].values[0]

            # Load the audio file
            audio = AudioSignal(wav_path)

            # Extract the tensor data and move it to the appropriate device
            # audio_tensor = audio.audio_data.to(device)

            # Encode the audio tensor
            codes = model.compress(audio, n_quantizers=n_quantizers)

            # Define the output file path
            output_file = os.path.join(output_dir, file_path + '.dac')

            codes.save(output_file)
            # Save the encoded codes
            # with open(output_file, 'wb') as f:
            #     torch.save(codes, f)

# def create_train_test_dataframes(sound_dict, wav_filenames, sound_list, features, wav_path, dac_folder):

#     df = pd.DataFrame.from_dict(generate_class_mapping(sound_dict, wav_filenames), orient="index", columns=["Class Name"]).reset_index(names="Audio ID")
#     df["Freesound ID"] = df["Audio ID"].apply(lambda x: x.split("_")[0])
#     df["WAV Path"] = df["Audio ID"].apply(lambda x: os.path.join(wav_path, x) + ".wav")
#     df["DAC Path"] = df["Audio ID"].apply(lambda x: x + ".dac")
    
#     sound_features = np.vstack([get_features(sound, features) for sound in sound_list])

#     for i in range(len(features)):
#         df[f"Param {i + 1}"] = sound_features[:, i]

#     train, test = train_test_split(df, test_size=0.2, stratify=df["Class Name"])

#     train.to_excel(os.path.join(dac_folder, "dac-train.xlsx"))
#     test.to_excel(os.path.join(dac_folder, "dac-val.xlsx"))

#     return train, test

def generate_train_test_dataframes(sound_dict, sound_list, wav_path, dac_folder, features):
    df = pd.DataFrame()
    mapping = generate_class_mapping(sound_dict)
    df["ID"] = [sound.id for sound in sound_list]
    df.set_index("ID", inplace=True)
    df["Name"] = [sound.name for sound in sound_list]
    df["WAV Path"] = [os.path.join(wav_path, f"{sound.id}.wav") for sound in sound_list]
    df["Full File Name"] = [os.path.join(f"{sound.id}.dac") for sound in sound_list]
    df["Class Name"] = [mapping[sound.id] for sound in sound_list]

    sound_features = np.vstack([get_features(sound, features) for sound in sound_list])

    for i in range(len(features)):
        df[f"Param {i + 1}"] = sound_features[:, i] 

    df2 = pd.DataFrame(columns=df.columns)
    for wav_filename in os.listdir(wav_path):
        sound_id = int(wav_filename.split('_')[0])
        df2.loc[wav_filename[:-4]] = df.loc[sound_id]
        df2.loc[wav_filename[:-4], "Full File Name"] = wav_filename[:-4] + ".dac"
        df2.loc[wav_filename[:-4], "WAV Path"] = df2.loc[wav_filename[:-4], "WAV Path"].replace(str(sound_id), wav_filename[:-4])

    from sklearn.preprocessing import MinMaxScaler
    df2.loc[:, df2.columns.str.startswith("Param")] = MinMaxScaler().fit_transform(df2.loc[:, df2.columns.str.startswith("Param")])
    df2.reset_index(inplace=True, names="ID")

    # Remove rows containing NaNs
    df2.dropna(inplace=True)
    train, test = train_test_split(df2, test_size=0.2, stratify=df2["Class Name"])

    train.reset_index(drop=True).to_excel(os.path.join(dac_folder, "dac-train.xlsx"))
    test.reset_index(drop=True).to_excel(os.path.join(dac_folder, "dac-val.xlsx"))
    return train, test

def create_sound_list(freesound_queries, freesound_client):
    sounds = sum([query_freesound(query['query'], query['filter'], freesound_client, query['num_results']) for query in freesound_queries],[])
    return sounds

def create_sound_dict(freesound_queries, freesound_client):
    sound_dict = {query["query"]: query_freesound(query['query'], query['filter'], freesound_client, query['num_results']) for query in freesound_queries}
    return sound_dict

def generate_class_mapping(sound_dict):
    id_to_class = dict()
    for class_name, sounds in sound_dict.items():
        for sound in sounds:
            id_to_class[sound.id] = class_name
    return id_to_class
    # filenames_to_class = dict()
    # for filename in wav_filenames:
    #     basename = os.path.basename(filename)[:-4]
    #     sound_id = int(os.path.basename(filename).split('_')[0])
    #     filenames_to_class[basename] = id_to_class[sound_id]
    # return filenames_to_class

def query_freesound(query, filter, freesound_client, num_results=10):
    """Queries freesound with the given query and filter values.
    If no filter is given, a default filter is added to only get sounds shorter than 30 seconds.
    """
    if filter is None:
        filter = 'duration:[0 TO 30]'  # Set default filter
    pager = freesound_client.text_search(
        query = query,
        filter = filter,
        fields = ','.join(FREESOUND_STORE_METADATA_FIELDS),
        group_by_pack = 1,
        page_size = num_results
    )
    return [sound for sound in pager]
    
def retrieve_sound_preview(sound, freesound_client, directory):
    """Download the high-quality OGG sound preview of a given Freesound sound object to the given directory.
    """
    return freesound.FSRequest.retrieve(
        sound.previews.preview_hq_ogg,
        freesound_client,
        os.path.join(directory, str(sound.id))
    )

def download_sound_previews(sounds, directory, freesound_client):
    """Download the high-quality OGG sound previews of a list of Freesound sound objects to the given directory.
    """
    for sound in tqdm(sounds, desc='Downloading sound previews'):
        retrieve_sound_preview(sound, freesound_client, directory)

def ogg_to_wav(ogg_folder, wav_folder, dur=5):
    total_length = 0
    wav_filenames = list()
    for filename in os.listdir(ogg_folder):
        ogg_path = os.path.join(ogg_folder, filename)
        
        # Change extension to WAV
        wav_path = os.path.join(wav_folder, filename)

        # Load OGG file using Essentia
        audio = es.MonoLoader(filename=ogg_path)()
        
        # Trim to first 5 seconds (Essentia loads audio as a NumPy array with a sample rate)
        sample_rate = 44100  # Default sample rate in Essentia

        total_length += len(audio) / sample_rate
        
        # Trim audio into 5s frames:
        for i in range(0, len(audio), 5 * sample_rate):
            trimmed_audio = audio[i:i + 5 * sample_rate]
            if len(trimmed_audio) < 5 * sample_rate:
                continue
            # Save as WAV. include i right before the extension
            path = wav_path + f"_{i}.wav"
            
            es.MonoWriter(filename=path, format='wav')(trimmed_audio)

            wav_filenames.append(path)
    return wav_filenames, total_length

def get_features(sound: freesound.FreesoundObject, features: list):
    """Get the features of a given sound object.
    """
    feat_vector = list()
    try:
        a = sound.get_analysis()
        feat_vector = list()
        for feature in features:
            feat_vector.append(eval(f"a.{feature}"))
    except:
        return [np.nan for f in features]
    return feat_vector
    


def force_mkdir(directory):
    """Force the creation of a directory by deleting it if it already exists.
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

FEATURES = ["lowlevel.pitch.mean",
        "rhythm.beats_count",
        "rhythm.onset_rate",
        "rhythm.beats_loudness.mean",
        "rhythm.beats_loudness.var",
        "sfx.logattacktime.mean",
        ]

def main():
    force_mkdir(OGG_PATH)
    force_mkdir(WAV_PATH)

    freesound_client = freesound.FreesoundClient()
    freesound_client.set_token(FREESOUND_API_KEY)

    sound_list = create_sound_list(FREESOUND_QUERIES, freesound_client)
    sound_dict = create_sound_dict(FREESOUND_QUERIES, freesound_client)
    

    download_sound_previews(sound_list, OGG_PATH, freesound_client)

    wav_filenames, total_length = ogg_to_wav(OGG_PATH, WAV_PATH)

    train, test = generate_train_test_dataframes(sound_dict, sound_list, WAV_PATH, DAC_PATH, FEATURES)

    train_path = os.path.join(DAC_PATH, "dac-train")
    val_path = os.path.join(DAC_PATH, "dac-val")

    force_mkdir(train_path)
    force_mkdir(val_path)

    encode_audio_files(train, train_path, model_sr='44khz', model_tag='latest')
    encode_audio_files(test, val_path, model_sr='44khz', model_tag='latest')

if __name__=="__main__":
    main()