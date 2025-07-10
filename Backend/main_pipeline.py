from speaker_diarisation import speaker_diarisation
from main_llm_workflow import main_LLM
from TTS_phase_3_offline import main_TTS
import sys
import os

def main_pipeline(audio_file_path, meeting_starttime, meeting_endtime):
    print(f"=== COMPREHENSIVE PIPELINE START ===")
    print(f"Audio file: {audio_file_path}")
    print(f"Start time: {meeting_starttime}")
    print(f"End time: {meeting_endtime}")
    
    # Check if audio file exists
    if not os.path.exists(audio_file_path):
        print(f"ERROR: Audio file '{audio_file_path}' not found", file=sys.stderr)
        sys.exit(1)
    
    file_size = os.path.getsize(audio_file_path)
    print(f"Audio file size: {file_size} bytes")
    
    if file_size == 0:
        print("ERROR: Audio file is empty", file=sys.stderr)
        sys.exit(1)
    
    try:
        print("\n=== PHASE 1: SPEAKER DIARISATION ===")
        print("Running speaker diarisation (keeping original speaker labels A, B, C)...")
        transcript_path, merged_output = speaker_diarisation(audio_file_path, 180)
        
        print(f"Speaker diarisation completed:")
        print(f"  - Transcript path: {transcript_path}")
        print(f"  - Merged output length: {len(merged_output) if merged_output else 0}")
        
        # Show sample of speakers found
        if merged_output:
            speakers_found = set()
            for segment in merged_output[:5]:
                lines = segment.strip().split('\n')
                if len(lines) >= 1:
                    header = lines[0]
                    if '|' in header:
                        speaker = header.split('|')[-1].strip()
                        speakers_found.add(speaker)
            print(f"  - Speakers identified: {sorted(list(speakers_found))}")
        
        if not merged_output:
            print("ERROR: merged_output is empty")
            sys.exit(1)
        
        print(f"\n=== PHASE 2: COMPREHENSIVE LLM ANALYSIS ===")
        os.makedirs('processed', exist_ok=True)
        
        print("Starting comprehensive analysis including:")
        print("  - High-quality topic extraction with semantic analysis")
        print("  - RAG-based point extraction with FAISS indexing") 
        print("  - Detailed speaker-specific analysis")
        print("  - Comprehensive document formatting")
        print("  - Professional podcast script generation")
        
        podcast_text = main_LLM(meeting_starttime, meeting_endtime, transcript_path, 0, 
                               'processed/Minutes_Echonote.docx', merged_output)
        
        print(f"Comprehensive LLM analysis completed")
        print(f"Generated summary length: {len(podcast_text)} characters")
        
        print(f"\n=== PHASE 3: TTS PROCESSING ===")
        print("Generating high-quality audio summary...")
        main_TTS(podcast_text, "processed/Podcast.wav")  # Changed from Meeting_Analysis_Audio.wav
        print("TTS processing completed")
        
        print(f"\n=== COMPREHENSIVE PIPELINE COMPLETED SUCCESSFULLY ===")
        print("Generated files:")
        print("  - processed/Comprehensive_Analysis_Report.txt (detailed analysis with speaker insights)")
        print("  - processed/Podcast_Script.txt (professional script)")
        print("  - processed/Podcast.wav (audio summary)")  # Updated filename
        
    except Exception as e:
        print(f"ERROR in comprehensive pipeline: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python main_pipeline.py <audio_file_path> <meeting_starttime> <meeting_endtime>")
        sys.exit(1)
    
    main_pipeline(sys.argv[1], sys.argv[2], sys.argv[3])
