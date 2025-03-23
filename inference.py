import torch
import time
from utils.utils import generate_mask, load_model, writeDACFile, sample_top_n
from dataloader.dataset import CustomDACDataset
from utils.utils import interpolate_vectors, breakpoints, breakpoints_classseq

import os
import yaml

from DACTransformer.RopeCondDACTransformer import RopeCondDACTransformer

import numpy as np
import matplotlib.pyplot as plt

import dac
import soundfile as sf

import mido
import threading
import queue
import sounddevice as sd


# Shared class index variable, updated by MIDI thread
current_class_idx = 0  # Default to the first class (update via MIDI)

def midi_listener(port_name, num_classes, n_params=6):
    """
    Listens for MIDI input and updates `current_class_idx` dynamically.
    
    Args:
        port_name (str): The name of the MIDI port to listen on.
        num_classes (int): Number of available classes.
    """
    global current_class_idx
    global params
    params = torch.zeros(n_params).to(device)
    params = params + .5

    try:
        with mido.open_input(port_name) as port:
            print(f"Listening for MIDI on {port_name}...")
            for msg in port:
                if msg.type == 'note_on':  # Use note number to pick a class
                    new_class = msg.note % num_classes  # Map MIDI notes to classes
                    current_class_idx = new_class
                    print(f"Updated class index: {current_class_idx}")
                
                elif msg.type == 'control_change':  # Use CC for a different mapping
                    new_param = msg.value
                    param_idx = msg.control%21
                    params[param_idx] = new_param/127
                    print(params)
                    print(f"Updated param {param_idx} via CC: {new_param}")

    except Exception as e:
        print(f"MIDI Error: {e}")

def generate_midi_controlled_cond(inference_steps, classes, param_count=1):
    """
    Generates a conditioning sequence where the class index is controlled by live MIDI input.
    
    Args:
        inference_steps (int): Number of time steps (frames) for inference.
        classes (list): List of class names (one-hot encoded).
        param_count (int): Number of continuous parameters (random walk).
    
    Returns:
        cond: A Tensor of shape (1, inference_steps, cond_size).
    """
    num_classes = len(classes)
    cond_size = num_classes + param_count

    # Prepare a buffer for (inference_steps, cond_size)
    cond = torch.zeros(inference_steps, cond_size)
    
    for i, p in enumerate(params):
        cond[0, num_classes + i] = p
    
    global current_class_idx
    cond[0, current_class_idx] = 1.0
    for t in range(1, inference_steps):

        # Copy previous step
        cond[t] = cond[t-1]

        # Reset the class portion to zero, then set the one-hot class
        cond[t, :num_classes] = 0.0
        cond[t, current_class_idx] = 1.0

    return cond.unsqueeze(0)

def inference(model, inference_cond, Ti_context_length, vocab_size, num_tokens, inference_steps, topn, fname, prev_tokens) :
    model.eval()
    with torch.no_grad():
        # print(Ti_context_length)
        mask = generate_mask(Ti_context_length, Ti_context_length).to(device)

        # pseudocódigo
        prev_context = prev_tokens[:, -Ti_context_length:, :]
        # y en la próxima llamada:
        input_data = prev_context

        #input_data = torch.randint(0, vocab_size, (1, Ti_context_length, num_tokens)).to(device)  # Smaller context window for inference
        # print("Inference cond: ", inference_cond.shape)
        #Extend the first conditional vector to cover the "input" which is of length Ti_context_length
        inference_cond = torch.cat([inference_cond[:, :1, :].repeat(1, Ti_context_length, 1), inference_cond], dim=1)
        # print("Inference cond: ", inference_cond.shape)
        predictions = []

        
        t0 = time.time()
        for i in range(inference_steps):  # 
            # print(input_data.shape, inference_cond.shape, mask.shape)

            if cond_size == 0:
                output = model(input_data, None, mask) # step through 
            else :
                # print(f'input_data.shape = {input_data.shape}, inference_cond[:, i:Ti_context_length+i, :].shape = {inference_cond[:, i:Ti_context_length+i, :].shape}, mask.shape = {mask.shape}')
                # print(inference_cond[:, i:Ti_context_length+i, :])
                output = model(input_data, inference_cond[:, i:Ti_context_length+i, :], mask) # step through

            # This takes the last vector of the sequence (the new predicted token stack) so has size(b,steps,4,1024)
            # This it takes the max across the last dimension (scores for each element of the vocabulary (for each of the 4 tokens))
            # .max returns a duple of tensors, the first are the max vals (one for each token) and the second are the
            #        indices in the range of the vocabulary size. 
            # THAT IS, the selected "best" tokens (one for each codebook) are taken independently
            ########################### next_token = output[:, -1, :, :].max(-1)[1]  # Greedy decoding for simplicity

            next_token = sample_top_n(output[:, -1, :, :],topn) # topn=1 would be the same as max in the comment line above    
            predictions.append(next_token)
            input_data = torch.cat([input_data, next_token.unsqueeze(1)], dim=1)[:, 1:]  # Slide window

        t1 = time.time()
        inf_time = t1-t0

        dacseq = torch.cat(predictions, dim=0).unsqueeze(0).transpose(1, 2)

        return dacseq
    
def generate_audio():
    with torch.no_grad():
        prev_tokens = torch.randint(0, vocab_size, (1, Ti_context_length, num_codebooks)).to(device)

        while True:
            print(f'current_class = {classes[current_class_idx]}')
            cond = generate_midi_controlled_cond(
                inference_steps, 
                classes, 
                param_count=6).to(device)  

            codeseq = inference(model, cond, Ti_context_length, vocab_size, num_codebooks, inference_steps, topn, "", prev_tokens)
            # print(f'codeseq shape = {codeseq.shape}')
            prev_tokens = codeseq.reshape(1, -1, num_codebooks)
            dac_file = dac.DACFile(
                codes=codeseq.cpu(),
                chunk_length=codeseq.shape[2],
                original_length=int(codeseq.shape[2] * 512),
                input_db=torch.tensor(-20),
                channels=1,
                sample_rate=44100,
                padding=True,
                dac_version='1.0.0'
            )
            audio_signal = dacmodel.decompress(dac_file)
            audio_data = audio_signal.samples.view(-1).numpy()
            
            # Enqueue audio data
            # Slice the audio_data into 4096 frames blocks
            audio_queue.put(audio_data[:blocksize])
            # for i in range(0, len(audio_data), blocksize):
            #     chunk = audio_data[i:i+blocksize]
            #     audio_queue.put(chunk)

# Audio callback function
def audio_callback(outdata, frames, time_info, status):
    if status:
        print(f"Status: {status}")
    try:
        chunk = audio_queue.get_nowait()
        print(chunk)
        print(frames)
        outdata[:len(chunk)] = chunk.reshape(-1, 1)
        if len(chunk) < frames:
            outdata[len(chunk):] = 0
    except queue.Empty:
        outdata.fill(0)

if __name__=='__main__':

    experiment_name= "mini_test_01" 
    checkpoint_dir = 'runs' + '/' + experiment_name  

    cptnum =  100
    DEVICE='cuda'
    gendur=20 
    topn=1 
    device = DEVICE
    
    paramfile = checkpoint_dir + '/' +  'params.yaml' 
    print(f"will use paramfile= {paramfile}") 
    
    # Load YAML file
    with open(paramfile, 'r') as file:
        params = yaml.safe_load(file)

    # Create an instance of the dataset
    data_dir = params['data_dir']
    data_frames =  params['data_frames']
    dataset = CustomDACDataset(data_dir=data_dir, metadata_excel=data_frames, transforms=None)
    classes = dataset.get_class_list()

    FEATURES = params['FEATURES']

    inference_steps=86*gendur  #86 frames per second
    
    TransformerClass =  globals().get(params['TransformerClass'])

    embed_size = params['model_size']

    fnamebase='out' + '.e' + str(embed_size) + '.l' + str(params['num_layers']) + '.h' + str(params['num_heads']) + '_chkpt_' + str(cptnum).zfill(4) 
    checkpoint_path = checkpoint_dir + '/' +  fnamebase  + '.pth' 

    if DEVICE == 'cuda' :
        torch.cuda.device_count()
        torch.cuda.get_device_properties(0).total_memory/1e9

        device = torch.device(DEVICE) # if the docker was started with --gpus all, then can choose here with cuda:0 (or cpu)
        torch.cuda.device_count()
        print(f'memeory on cuda 0 is  {torch.cuda.get_device_properties(0).total_memory/1e9}')
    else :
        device=DEVICE
    device

    #Load the stored model
    model, _, Ti_context_length, vocab_size, num_codebooks, cond_size = load_model(checkpoint_path,  TransformerClass, DEVICE)

    print(f'Mode loaded, context_length (Ti_context_length) = {Ti_context_length}')
    # Count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {num_params}')

    model.to(device)

    dacmodel_path = dac.utils.download(model_type="44khz") 
    print(f'The DAC decoder is in {dacmodel_path}')
    with torch.no_grad():
        dacmodel = dac.DAC.load(dacmodel_path)

        dacmodel.to(device); #wanna see the model? remove the semicolon
        dacmodel.eval();  # need to be "in eval mode" in order to set the number of quantizers

    # Initialize audio queue
    try:
        del audio_queue
    except:
        pass
    audio_queue = queue.Queue()

    # Print available midi ports
    port = mido.get_input_names()[1]
    # Start MIDI listener thread
    midi_thread = threading.Thread(target=midi_listener, args=(port, len(classes)))  # Replace "YourMIDIport" with actual port
    # midi_thread.daemon = True
    midi_thread.start()

    dur = 1
    blocksize = int(44100 * dur)
    inference_steps = int(86 * dur)
    Ti_context_length = int(86 * dur)

    # Start audio stream
    samplerate = 44100
    stream = sd.OutputStream(
        samplerate=samplerate,
        channels=1,
        blocksize=blocksize,
        callback=audio_callback
    )
    stream.start()

    # Start audio generation in a separate thread
    audio_thread = threading.Thread(target=generate_audio)
    # audio_thread.daemon = True
    audio_thread.start()






