import os
from pydub import AudioSegment

def convert_flac_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path, format="flac")
    audio.export(output_path, format="wav")

def convert_all_flac_to_wav(directory_path):
    output_directory = os.path.join("", "/Users/bhanutejavaravenkata/desktop/language_data/train")
    os.makedirs(output_directory, exist_ok=True)

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".flac"):
            flac_path = os.path.join(directory_path, file_name)
            wav_path = os.path.join(output_directory, file_name.replace(".flac", ".wav"))

            print(f"Converting {file_name} to {wav_path}")
            convert_flac_to_wav(flac_path, wav_path)


input_directory = '/Users/bhanutejavaravenkata/desktop/Main_dataset/train/train'
convert_all_flac_to_wav(input_directory)
print("Conversion complete!")
