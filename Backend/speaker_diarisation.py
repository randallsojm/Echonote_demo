import whisper
from pyannote.audio import Pipeline
from utils import diarize_text
import torch
import os
import sys

def seconds_to_minutes(seconds):
    minutes = int(seconds) // 60
    seconds = int(seconds) % 60
    return f"{minutes}.{seconds:02}"

def chunk_by_time_intervals(segments, interval_seconds):
    chunks = []
    current_chunk = []
    chunk_start_time = None
    
    for i, (seg, speaker, transcript) in enumerate(segments):
        start_time = seg.start
        
        if chunk_start_time is None:
            chunk_start_time = start_time

        if start_time - chunk_start_time >= interval_seconds:
            chunks.append(current_chunk)
            current_chunk = []
            chunk_start_time = start_time

        current_chunk.append((seg, speaker, transcript))

    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def speaker_diarisation(audio_file_path, chunk_interval_seconds):
    print(f"=== SPEAKER DIARISATION START ===")
    print(f"Audio file: {audio_file_path}")
    
    if not os.path.exists(audio_file_path):
        print(f"ERROR: Audio file not found")
        return None, None
    
    dataset = audio_file_path

    try:
        print("Loading Whisper model...")
        model = whisper.load_model("whisper/tiny.en.pt")
        print("Whisper model loaded successfully.")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device, dtype=torch.float32)

    try:
        print("Loading pyannote pipeline...")
        pipeline = Pipeline.from_pretrained("config.yaml")
        print("pyannote pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading pyannote pipeline: {e}")
        return None, None

    try:
        print("Running ASR transcription...")
        asr_result = model.transcribe(dataset)
        
        print("Running speaker diarisation...")
        diarization_result = pipeline(dataset)
        
        print("Running diarize_text...")
        final_result = diarize_text(asr_result, diarization_result)
        
        if not final_result:
            print("ERROR: diarize_text returned empty result!")
            return None, None
            
    except Exception as e:
        print(f"Error during diarisation pipeline: {e}")
        return None, None

    # Keep original speaker labels (A, B, C, etc.) - NO SPEAKER IDENTIFICATION
    merged_output = []
    segments_for_chunking = []
    
    print(f"Processing {len(final_result)} segments - keeping original speaker labels...")
    
    for seg, speaker, transcript in final_result:
        # Convert SPEAKER_00 -> A, SPEAKER_01 -> B, etc.
        if speaker.startswith('SPEAKER_'):
            speaker_num = int(speaker.split('_')[1])
            speaker_label = chr(65 + speaker_num)  # A, B, C, etc.
        else:
            speaker_label = speaker
            
        segments_for_chunking.append((seg, speaker_label, transcript))

    if not segments_for_chunking:
        print("ERROR: No segments for chunking!")
        return None, None

    # Perform chunking
    chunked_segments = chunk_by_time_intervals(segments_for_chunking, chunk_interval_seconds)
    
    if not chunked_segments:
        print("ERROR: No chunked segments created!")
        return None, None

    for chunk in chunked_segments:
        prev_speaker = None
        prev_start = None
        prev_end = None
        prev_transcript = ""
                
        for seg, speaker, transcript in chunk:
            timestart = seconds_to_minutes(seg.start)
            timeend = seconds_to_minutes(seg.end)
            
            if speaker == prev_speaker and prev_end == timestart:
                prev_end = timeend
                prev_transcript += " " + transcript
            else:
                if prev_speaker is not None:
                    merged_line = f"{prev_start} --> {prev_end} | {prev_speaker}\n{prev_transcript}\n\n"
                    merged_output.append(merged_line)
                    
                prev_speaker = speaker
                prev_start = timestart
                prev_end = timeend
                prev_transcript = transcript

        if prev_speaker is not None:
            merged_line = f"{prev_start} --> {prev_end} | {prev_speaker}\n{prev_transcript}\n\n"
            merged_output.append(merged_line)

    if not merged_output:
        print("ERROR: merged_output is empty!")
        return None, None
                
    file_path = "Processed/Transcript.txt"
    try:
        with open(file_path, "w", encoding='utf-8') as file:
            file.write("".join(merged_output))
        print(f"Transcript written to {file_path}")
    except Exception as e:
        print(f"Error writing transcript file: {e}")
        return None, None

    print("=== SPEAKER DIARISATION COMPLETED ===")
    return file_path, merged_output
