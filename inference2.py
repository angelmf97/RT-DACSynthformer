import torch
import time
import numpy as np
import threading
import sounddevice as sd
import mido
import os
import yaml
import dac
from utils.utils import generate_mask, load_model, sample_top_n
from dataloader.dataset import CustomDACDataset
from DACTransformer.RopeCondDACTransformer import RopeCondDACTransformer
import tkinter as tk

# Global variables for MIDI control
current_class_idx = 0
params = None  # This will hold continuous control parameters

# --- Ring Buffer Implementation ---
class RingBuffer:
    def __init__(self, size):
        self.buffer = np.zeros(size, dtype=np.float32)
        self.size = size
        self.write_ptr = 0
        self.read_ptr = 0
        self.lock = threading.Lock()

    def write(self, data):
        with self.lock:
            for sample in data:
                self.buffer[self.write_ptr] = sample
                self.write_ptr = (self.write_ptr + 1) % self.size
                if self.write_ptr == self.read_ptr:
                    self.read_ptr = (self.read_ptr + 1) % self.size

    def read(self, n):
        with self.lock:
            available = self.available_data()
            n = min(n, available)
            out = np.zeros(n, dtype=np.float32)
            for i in range(n):
                out[i] = self.buffer[self.read_ptr]
                self.read_ptr = (self.read_ptr + 1) % self.size
            return out

    def available_data(self):
        if self.write_ptr >= self.read_ptr:
            return self.write_ptr - self.read_ptr
        else:
            return self.size - (self.read_ptr - self.write_ptr)

# Global ring buffer sized to store about 10 seconds of audio
BUFFER_SIZE = 44100 * 10
ring_buffer = RingBuffer(BUFFER_SIZE)

# --- MIDI Listener ---
def midi_listener(port_name, num_classes, n_params=6):
    global current_class_idx, params
    params = torch.zeros(n_params).to(device)

    try:
        with mido.open_input(port_name) as port:
            print(f"Listening for MIDI on {port_name}...")
            for msg in port:
                if msg.type == 'note_on':
                    current_class_idx = msg.note % num_classes
                    print(f"Updated class index: {current_class_idx}")
                elif msg.type == 'control_change':
                    param_idx = msg.control % 21
                    params[param_idx] = msg.value / 127.0
                    print(f"Updated param {param_idx}: {params[param_idx]}")
    except Exception as e:
        print(f"MIDI Error: {e}")
        raise(e)

# --- Conditioning Sequence Generation ---
def generate_midi_controlled_cond(inference_steps, classes, param_count=6):
    num_classes = len(classes)
    cond_size = num_classes + param_count
    cond = torch.zeros(inference_steps, cond_size)
    # Set continuous parameters from MIDI CC messages
    global params
    for i, p in enumerate(params):
        cond[0, num_classes + i] = p
    global current_class_idx
    cond[0, :num_classes] = 0.0
    cond[0, current_class_idx] = 1.0
    for t in range(1, inference_steps):
        cond[t] = cond[t - 1]
        cond[t, :num_classes] = 0.0
        cond[t, current_class_idx] = 1.0
    return cond.unsqueeze(0)

# --- Inference Function ---
def inference(model, inference_cond, Ti_context_length, vocab_size, num_tokens,
              inference_steps, topn, prev_tokens):
    """
    Returns a tensor of shape [1, inference_steps, num_tokens].
      e.g. [1, 86, 4] if inference_steps=86 and num_tokens=4.
    """
    model.eval()
    with torch.no_grad():
        # Generate the attention mask for the context length
        mask = generate_mask(Ti_context_length, Ti_context_length).to(device)

        # Extract the last Ti_context_length tokens to serve as input context
        prev_context = prev_tokens[:, -Ti_context_length:, :]  # shape [1, Ti_context_length, 4]
        
        input_data = prev_context


        # Pad the conditioning so it aligns with each step in the input
        # shape: [1, Ti_context_length + inference_steps, cond_size]
        inference_cond = torch.cat(
            [inference_cond[:, :1, :].repeat(1, Ti_context_length, 1), inference_cond],
            dim=1
        )

        # We'll store each predicted token in a list, then stack.
        predictions = []
        for i in range(inference_steps):
            # Pass input_data plus the relevant slice of conditioning
            if cond_size == 0:
                output = model(input_data, None, mask)
            else:
                cond_slice = inference_cond[:, i : Ti_context_length + i, :]
                output = model(input_data, cond_slice, mask)
            
            # 'output' shape: [batch=1, seq=Ti_context_length, codebooks, vocab_size]
            # We want the last time step: output[:, -1, :, :] => [1, codebooks, vocab_size]
            # Then pick a token index for each codebook
            next_token = sample_top_n(output[:, -1, :, :], topn)  # shape [1, codebooks]
            predictions.append(next_token)

            # Slide the input window by appending next_token
            # input_data shape was [1, Ti_context_length, codebooks]
            # so after cat => [1, Ti_context_length+1, codebooks], then we drop the first step
            input_data = torch.cat([input_data, next_token.unsqueeze(1)], dim=1)[:, 1:]
        
        # Now convert `predictions` (a list of length inference_steps)
        # into a single tensor: [1, inference_steps, codebooks].
        # Each item is shape [1, codebooks].
        # torch.cat(predictions, dim=0) => shape [inference_steps, 1, codebooks]
        # We'll squeeze dim=1 and re-add batch=1 in front:
        codeseq = torch.cat(predictions, dim=0)   # [inference_steps, 1, codebooks]
        codeseq = codeseq.squeeze(1)              # [inference_steps, codebooks]
        codeseq = codeseq.unsqueeze(0)            # [1, inference_steps, codebooks]
        
        return codeseq


# --- Audio Generation ---
def generate_audio():
    global ring_buffer
    with torch.no_grad():
        # Initialize with random tokens for 3 seconds of context
        
        while True:
            prev_tokens = torch.randint(0, vocab_size, (1, Ti_context_length, num_codebooks)).to(device)
            cond = generate_midi_controlled_cond(inference_steps, classes, param_count=6).to(device)
            
            # Now inference(...) returns shape [1, 1_second_length, codebooks]
            codeseq = inference(model, cond, Ti_context_length, vocab_size,
                                num_codebooks, inference_steps, topn, prev_tokens)
            
            # # Append new tokens to previous tokens, then keep only the last 3s
            # prev_tokens = torch.cat([prev_tokens, codeseq], dim=1)  # cat along time dimension
            # prev_tokens = prev_tokens[:, -Ti_context_length:, :]    # keep last 3 seconds

            # For DAC decompression, we usually need shape [1, codebooks, length].
            # So let's permute codeseq to match. If codeseq is [1, length, codebooks],
            # we do:
            codeseq_dac = codeseq.permute(0, 2, 1)  # => [1, codebooks, length]
            
            dac_file = dac.DACFile(
                codes=codeseq_dac.cpu(),
                chunk_length=codeseq_dac.shape[2],
                original_length=int(codeseq_dac.shape[2] * 512),
                input_db=torch.tensor(-20),
                channels=1,
                sample_rate=44100,
                padding=True,
                dac_version='1.0.0'
            )
            audio_signal = dacmodel.decompress(dac_file)
            audio_data = audio_signal.samples.view(-1).numpy()

            ring_buffer.write(audio_data)


def create_gui():
    """
    Create a Tkinter window that shows `current_class_idx` and the values
    of `params` in real time.
    """
    global current_class_idx, params
    root = tk.Tk()
    root.title("Real-Time Visualization")

    # Label to show current_class_idx
    idx_label = tk.Label(root, text=f"Current class: {classes[current_class_idx]}", font=("Arial", 14))
    idx_label.pack(pady=5)

    # A set of labels to display each element of `params`.
    param_labels = []
    for i, param_val in enumerate(params):
        lbl = tk.Label(root, text=f"Param {i + 1}: {param_val:.2f}", font=("Arial", 12))
        lbl.pack()
        param_labels.append(lbl)

    def update_gui():
        """
        Refresh the window with new values from the global variables.
        This method re-schedules itself every 100ms.
        """
        print(params)
        # Update the current_class_idx text
        idx_label.config(text=f"Current class: {classes[current_class_idx]}")

        # Update each param label
        for i, val in enumerate(params):
            param_labels[i].config(text=f"{FEATURES[i]}: {val:.2f}")

        # Schedule the next update in 100ms
        root.after(100, update_gui)

    # Kick off the periodic GUI update
    update_gui()

    # Start the Tkinter event loop. This will block until the window is closed.
    root.mainloop()

# --- Audio Callback ---
def audio_callback(outdata, frames, time_info, status):
    if status:
        print(f"Status: {status}")
    data = ring_buffer.read(frames)
    if len(data) < frames:
        outdata[:len(data)] = data.reshape(-1, 1)
        outdata[len(data):] = 0
    else:
        outdata[:] = data.reshape(-1, 1)

# --- Main Execution ---
if __name__ == '__main__':
    experiment_name = "mini_test_02"
    checkpoint_dir = os.path.join('runs', experiment_name)
    cptnum = 250
    DEVICE = 'cuda'
    topn = 1
    device = DEVICE

    paramfile = os.path.join(checkpoint_dir, 'params.yaml')
    print(f"Using paramfile= {paramfile}")
    with open(paramfile, 'r') as file:
        params_yaml = yaml.safe_load(file)

    data_dir = params_yaml['data_dir']
    data_frames = params_yaml['data_frames']
    dataset = CustomDACDataset(data_dir=data_dir, metadata_excel=data_frames, transforms=None)
    classes = dataset.get_class_list()
    FEATURES = params_yaml['FEATURES']

    # Define durations:
    # - Context: 3 seconds of audio (used for the transformer context)
    # - Generation: 1 second of new audio per inference
    context_duration = 3.0      # seconds
    generation_duration = 1.0   # seconds

    callback_blocksize = 1024   # frames per audio callback (adjust for latency)
    inference_steps = int(86 * generation_duration)
    Ti_context_length = int(86 * context_duration)

    TransformerClass = globals().get(params_yaml['TransformerClass'])
    embed_size = params_yaml['model_size']
    fnamebase = f'out.e{embed_size}.l{params_yaml["num_layers"]}.h{params_yaml["num_heads"]}_chkpt_{str(cptnum).zfill(4)}'
    checkpoint_path = os.path.join(checkpoint_dir, fnamebase + '.pth')

    if DEVICE == 'cuda':
        torch.cuda.device_count()
        device = torch.device(DEVICE)
        print(f'Memory on cuda 0: {torch.cuda.get_device_properties(0).total_memory / 1e9} GB')
    else:
        device = DEVICE

    model, _, Ti_context_length_loaded, vocab_size, num_codebooks, cond_size = load_model(checkpoint_path, TransformerClass, DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    model.to(device)

    dacmodel_path = dac.utils.download(model_type="44khz")
    print(f'The DAC decoder is in {dacmodel_path}')
    with torch.no_grad():
        dacmodel = dac.DAC.load(dacmodel_path)
        dacmodel.to(device)
        dacmodel.eval()

    # Start the MIDI listener thread.
    midi_ports = mido.get_input_names()
    if not midi_ports:
        print("No MIDI ports found.")
        exit(1)

    print("Available MIDI ports:", midi_ports)
    chosen_port = input(f"Choose a MIDI port [number from {0} to {len(midi_ports)}]:")  # Adjust if needed
    midi_port = midi_ports[1]  # Adjust if needed
    midi_thread = threading.Thread(target=midi_listener, args=(midi_port, len(classes)))
    midi_thread.daemon = True
    midi_thread.start()

    # Start the audio generation thread.
    audio_gen_thread = threading.Thread(target=generate_audio)
    audio_gen_thread.daemon = True
    audio_gen_thread.start()

    # Pre-buffer some audio before playback.
    prebuffer_seconds = 1.0
    prebuffer_samples = int(44100 * prebuffer_seconds)
    print("Pre-buffering audio...")
    while ring_buffer.available_data() < prebuffer_samples:
        time.sleep(0.01)
    print("Pre-buffering complete.")

    # Start the audio output stream.
    samplerate = 44100
    stream = sd.OutputStream(
        samplerate=samplerate,
        channels=1,
        blocksize=callback_blocksize,
        callback=audio_callback
    )
    stream.start()
    print("Audio streaming started. Press Ctrl+C to stop.")
    
    create_gui()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping...")
        stream.stop()
