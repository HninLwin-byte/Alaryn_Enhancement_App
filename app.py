#WORKING CODE with resemble ai
import streamlit as st
from wavlm.WavLM import WavLM, WavLMConfig
from hifigan.models import Generator as HiFiGAN
from hifigan.utils import AttrDict
from matcher import KNeighborsVC
from pathlib import Path
import json
import torch
import torchaudio
import io
import numpy as np
from audio_recorder_streamlit import audio_recorder

def hifigan_wavlm(pretrained=True, progress=True, prematched=True, device='cuda') -> HiFiGAN:
    """ Load pretrained HiFiGAN trained to vocode WavLM features. Optionally use weights trained on `prematched` data. """
    cp = Path.cwd().absolute()

    with open(cp / 'hifigan' / 'config_v1_wavlm.json') as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    device = torch.device(device)

    generator = HiFiGAN(h).to(device)

    if pretrained:
        if prematched:
            model_path = "/content/drive/MyDrive/g_02500000.pt"
        else:
            model_path = "/content/drive/MyDrive/do_00000040.pt"

        state_dict_g = torch.load(model_path, map_location=device)
        generator.load_state_dict(state_dict_g['generator'])

    generator.eval()
    generator.remove_weight_norm()
    print(f"[HiFiGAN] Generator loaded with {sum([p.numel() for p in generator.parameters()]):,d} parameters.")
    return generator, h

def wavlm_large(pretrained=True, progress=True, device='cuda') -> WavLM:
    if torch.cuda.is_available() == False:
        if str(device) != 'cpu':
            logging.warning(f"Overriding device {device} to cpu since no GPU is available.")
            device = 'cpu'
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt",
        map_location=device,
        progress=progress
    )
    cfg = WavLMConfig(checkpoint['cfg'])
    device = torch.device(device)
    model = WavLM(cfg)
    if pretrained:
        model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    print(f"WavLM-Large loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters.")
    return model

def knn_vc(pretrained=True, progress=True, prematched=True, device='cuda') -> KNeighborsVC:
    hifigan, hifigan_cfg = hifigan_wavlm(pretrained, progress, prematched, device)
    wavlm = wavlm_large(pretrained, progress, device)

    knnvc = KNeighborsVC(wavlm, hifigan, hifigan_cfg, device)
    return knnvc



def main():
    st.title("Alaryngeal Speech Enhancement App")

    # Let users choose between uploading or recording
    upload_or_record = st.radio("Do you want to upload a file or record your speech?", ("Upload", "Record"))

    src_audio_data = None
    # if upload_or_record == "Upload":
    #     src_wav_file = st.file_uploader("Upload Source WAV File", type=["wav"])
    #     if src_wav_file is not None:
    #         src_audio_data = src_wav_file.read()
    # elif upload_or_record == "Record":
    #     # Use the audio_recorder component for recording
    #     src_audio_data = audio_recorder()

        
    if upload_or_record == "Upload":
      src_wav_file = st.file_uploader("Upload Source WAV File", type=["wav"])
      if src_wav_file is not None:
          src_wav_path = "temp.wav"
          with open(src_wav_path, "wb") as f:
              f.write(src_wav_file.getvalue())

         
          knnvc_model = knn_vc(pretrained=True, progress=True, prematched=True, device='cuda')

          # Get features from the source WAV file
          query_seq = knnvc_model.get_features(src_wav_path)

          # Load reference file from the local machine
          reference_file_path = "/content/drive/MyDrive/con_layer22_data/final_features.pt"
          loaded_data = torch.load(reference_file_path)

          # Match query sequence with loaded data
          out_wav = knnvc_model.match(query_seq, loaded_data, topk=20)

          # Convert the tensor to a NumPy array
          out_wav_np = out_wav.detach().numpy()

          # Display audio in Streamlit with sample rate specified
          st.audio(out_wav_np, format='audio/wav', sample_rate=16000)

          # Convert the NumPy array to bytes for downloading
          out_wav_bytes = io.BytesIO()
          np.save(out_wav_bytes, out_wav_np, allow_pickle=False)

          # Add a download button for the generated WAV file
          st.download_button(label="Download Generated WAV", data=out_wav_bytes.getvalue(), file_name='generated_audio.wav', mime='audio/wav')

    elif upload_or_record == "Record":
         src_audio_data = audio_recorder()
         if src_audio_data:
            st.audio(src_audio_data, format="audio/wav")

         knnvc_model = knn_vc(pretrained=True, progress=True, prematched=True, device='cuda')

          # Get features from the source WAV file
         query_seq = knnvc_model.get_features(src_wav_path)

          # Load reference file from the local machine
         reference_file_path = "/content/drive/MyDrive/con_layer22_data/final_features.pt"
         loaded_data = torch.load(reference_file_path)

          # Match query sequence with loaded data
         out_wav = knnvc_model.match(query_seq, loaded_data, topk=20)

          # Convert the tensor to a NumPy array
         out_wav_np = out_wav.detach().numpy()

          # Display audio in Streamlit with sample rate specified
         st.audio(out_wav_np, format='audio/wav', sample_rate=16000)

          # Convert the NumPy array to bytes for downloading
         out_wav_bytes = io.BytesIO()
         np.save(out_wav_bytes, out_wav_np, allow_pickle=False)

          # Add a download button for the generated WAV file
         st.download_button(label="Download Generated WAV", data=out_wav_bytes.getvalue(), file_name='generated_audio.wav', mime='audio/wav')


if __name__ == "__main__":
    main()
