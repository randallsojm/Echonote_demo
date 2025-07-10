import sys
import os
import json
from typing import List, Dict, Optional
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import using LLM-based topic extraction
from topic_extraction import EnhancedLLMRAGSummarizer
from rag_point_extraction import RAGPointExtractor
from speaker_analysis import SpeakerAnalyzer
from document_formatter import DocumentFormatter
from podcast_generator import PodcastScriptGenerator

class CompleteLLMWorkflow:
    def __init__(self):
        print("Initializing complete LLM workflow with Llama 3.1 topic extraction...")
        self.topic_extractor = EnhancedLLMRAGSummarizer()
        self.rag_extractor = RAGPointExtractor()
        self.speaker_analyzer = SpeakerAnalyzer()
        self.document_formatter = DocumentFormatter()
        self.podcast_generator = PodcastScriptGenerator()
        self.executor = ThreadPoolExecutor(max_workers=4)
        print("Complete workflow with LLM topic extraction initialized")
    
    async def run_complete_analysis(self, 
                                  merged_output: List[str],
                                  meeting_starttime: str = None,
                                  meeting_endtime: str = None) -> str:
        """Run complete analysis with LLM-based topic extraction"""
        
        start_time = time.time()
        print(f"=== COMPLETE WORKFLOW WITH LLM START ({time.strftime('%H:%M:%S')}) ===")
        print(f"Processing {len(merged_output)} segments with Llama 3.1 topic extraction")
        
        try:
            # Step 1: LLM Topic Extraction
            print(f"\n--- STEP 1/5: LLM TOPIC EXTRACTION ({time.strftime('%H:%M:%S')}) ---")
            step_start = time.time()
            
            topics_task = asyncio.create_task(
                self._async_llm_topic_extraction(merged_output)
            )
            
            # Step 2: Speaker Analysis (Parallel)
            print(f"\n--- STEP 2/5: SPEAKER ANALYSIS ({time.strftime('%H:%M:%S')}) ---")
            speaker_task = asyncio.create_task(
                self._async_speaker_analysis(merged_output)
            )
            
            # Step 3: FAISS Preparation (Parallel)
            faiss_task = asyncio.create_task(
                self._async_faiss_preparation(merged_output)
            )
            
            # Wait for all parallel tasks
            raw_topics, speaker_analysis, faiss_ready = await asyncio.gather(
                topics_task, speaker_task, faiss_task
            )
            
            print(f"[OK] Parallel processing completed in {time.time() - step_start:.1f}s")
            print(f"  LLM Topics extracted: {len(raw_topics)}")
            print(f"  Speakers analyzed: {len(speaker_analysis.get('individual_insights', {}))}")
            
            # Show LLM-extracted topics
            for i, topic in enumerate(raw_topics[:5], 1):
                topic_name = topic.get('topic', 'Unknown')
                category = topic.get('category', 'General')
                confidence = topic.get('confidence', 0.0)
                print(f"    {i}. {topic_name} [{category}] (confidence: {confidence:.2f})")
            
            # Step 4: LLM Deduplication & RAG Analysis
            print(f"\n--- STEP 4/5: LLM DEDUPLICATION & RAG ANALYSIS ({time.strftime('%H:%M:%S')}) ---")
            step_start = time.time()
            
            # LLM-based deduplication
            deduplicated_topics = await self._async_llm_deduplicate_topics(raw_topics)
            print(f"LLM deduplicated to {len(deduplicated_topics)} topics")

            # --- NEW: Generate comprehensive LLM+RAG summary ---
            print(f"Generating comprehensive LLM+RAG summary...")
            loop = asyncio.get_event_loop()
            comprehensive_summary = await loop.run_in_executor(
                self.executor,
                self.topic_extractor.generate_comprehensive_summary,
                deduplicated_topics
            )
            print(f"[OK] LLM+RAG summary generated.")

            # Run RAG analysis for all topics in parallel
            rag_tasks = [
                self._async_rag_for_topic(topic_data) 
                for topic_data in deduplicated_topics
            ]
            rag_results = await asyncio.gather(*rag_tasks)
            print(f"[OK] RAG analysis completed in {time.time() - step_start:.1f}s")
            
            # Step 5: Document Generation
            print(f"\n--- STEP 5/5: DOCUMENT GENERATION ({time.strftime('%H:%M:%S')}) ---")
            step_start = time.time()
            
            # Generate comprehensive document with speaker analysis
            doc_task = asyncio.create_task(
                self._async_comprehensive_document_generation(
                    deduplicated_topics, rag_results, speaker_analysis, 
                    meeting_starttime, meeting_endtime
                )
            )
            
            # Generate podcast script
            podcast_task = asyncio.create_task(
                self._async_podcast_generation(deduplicated_topics, rag_results, speaker_analysis)
            )
            
            comprehensive_document, podcast_script = await asyncio.gather(doc_task, podcast_task)
            
            print(f"[OK] Documents generated in {time.time() - step_start:.1f}s")
            
            # Save Files with EXPECTED NAMES
            print(f"\n--- SAVING FILES ({time.strftime('%H:%M:%S')}) ---")
            step_start = time.time()
            
            save_tasks = [
                # Save with the names you expect
                self._async_save_file("processed/Minutes_Echonote.txt", comprehensive_document),
                self._async_save_file("processed/podcast.txt", podcast_script),
                self._async_save_json("processed/Analysis_Data.json", {
                    'topics_count': len(deduplicated_topics),
                    'topics': deduplicated_topics,
                    'speaker_analysis': speaker_analysis,
                    'rag_results_count': len(rag_results),
                    'processing_time': time.time() - start_time,
                    'extraction_method': 'llama3.1_8b'
                }),
                # --- NEW: Save comprehensive LLM+RAG summary ---
                self._async_save_json("processed/Comprehensive_Summary.json", comprehensive_summary)
            ]
            
            await asyncio.gather(*save_tasks)
            print(f"[OK] Files saved in {time.time() - step_start:.1f}s")
            
            # --- NEW: Load comprehensive summary and pass to formatter ---
            summary_path = "processed/Comprehensive_Summary.json"
            loaded_comprehensive_summary = None
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    loaded_comprehensive_summary = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load comprehensive summary for Markdown report: {e}")

            # Generate Markdown report again with summary if available
            if loaded_comprehensive_summary:
                comprehensive_document = self.document_formatter.format_comprehensive_analysis(
                    deduplicated_topics, speaker_analysis, rag_results, 
                    {
                        'start_time': meeting_starttime,
                        'end_time': meeting_endtime,
                        'participants': len(speaker_analysis.get('individual_insights', {})),
                        'duration': f"{meeting_starttime} - {meeting_endtime}",
                        'extraction_method': 'Llama 3.1 8B LLM'
                    },
                    loaded_comprehensive_summary
                )
                # Overwrite the Markdown file with the richer version
                await self._async_save_file("processed/Minutes_Echonote.txt", comprehensive_document)
            
            # Show what files were created
            print("Files created:")
            print("  - processed/Minutes_Echonote.txt (comprehensive analysis)")
            print("  - processed/podcast.txt (podcast script)")
            print("  - processed/Analysis_Data.json (raw data)")
            
            # Generate TTS summary with speaker insights
            summary = self._generate_complete_summary(
                deduplicated_topics, speaker_analysis, rag_results, time.time() - start_time
            )
            
            total_time = time.time() - start_time
            print(f"\n=== COMPLETE WORKFLOW WITH LLM FINISHED ===")
            print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            
            return summary
            
        except Exception as e:
            error_msg = f"Complete LLM workflow failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg
    
    async def _async_llm_topic_extraction(self, merged_output: List[str]) -> List[Dict]:
        """Extract topics using LLM asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.topic_extractor.extract_topics_from_transcript, 
            merged_output
        )
    
    async def _async_llm_deduplicate_topics(self, topics: List[Dict]) -> List[Dict]:
        """Deduplicate topics using LLM asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.topic_extractor.deduplicate_topics,
            topics
        )
    
    async def _async_speaker_analysis(self, merged_output: List[str]) -> Dict:
        """Analyze speakers asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.speaker_analyzer.analyze_speakers,
            merged_output
        )
    
    async def _async_faiss_preparation(self, merged_output: List[str]) -> bool:
        """Prepare FAISS index asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self.rag_extractor.build_faiss_index,
            merged_output
        )
        return True
    
    async def _async_rag_for_topic(self, topic_data: Dict) -> Dict:
        """Run RAG analysis for a single topic asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.rag_extractor.extract_points_for_topic,
            topic_data
        )
    
    async def _async_comprehensive_document_generation(self, topics: List[Dict], rag_results: List[Dict], 
                                                     speaker_analysis: Dict, start_time: str, end_time: str) -> str:
        """Generate comprehensive document with speaker analysis"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._generate_comprehensive_document,
            topics, rag_results, speaker_analysis, start_time, end_time
        )
    
    async def _async_podcast_generation(self, topics: List[Dict], rag_results: List[Dict], 
                                      speaker_analysis: Dict) -> str:
        """Generate podcast script with speaker insights"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._generate_comprehensive_podcast,
            topics, rag_results, speaker_analysis
        )
    
    def _generate_comprehensive_document(self, topics: List[Dict], rag_results: List[Dict], 
                                       speaker_analysis: Dict, start_time: str, end_time: str) -> str:
        """Generate document with full speaker analysis"""
        meeting_metadata = {
            'start_time': start_time,
            'end_time': end_time,
            'participants': len(speaker_analysis.get('individual_insights', {})),
            'duration': f"{start_time} - {end_time}",
            'extraction_method': 'Llama 3.1 8B LLM'
        }
        
        return self.document_formatter.format_comprehensive_analysis(
            topics, speaker_analysis, rag_results, meeting_metadata
        )
    
    def _generate_comprehensive_podcast(self, topics: List[Dict], rag_results: List[Dict], 
                                      speaker_analysis: Dict) -> str:
        """Generate podcast script with speaker insights"""
        # Use the existing podcast generator
        base_script = self.podcast_generator.generate_podcast_script(
            "", topics, speaker_analysis, style="conversational"
        )
        
        return base_script
    
    async def _async_save_file(self, filepath: str, content: str):
        """Save file asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self._save_text_file, filepath, content)
    
    async def _async_save_json(self, filepath: str, data: Dict):
        """Save JSON file asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self._save_json_file, filepath, data)
    
    def _save_text_file(self, filepath: str, content: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Saved: {filepath}")
    
    def _save_json_file(self, filepath: str, data: Dict):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Saved: {filepath}")
    
    def _generate_complete_summary(self, topics: List[Dict], speaker_analysis: Dict, 
                                 rag_results: List[Dict], processing_time: float) -> str:
        """Generate complete summary including speaker insights for TTS"""
        individual_insights = speaker_analysis.get('individual_insights', {})
        num_speakers = len(individual_insights)
        
        # Get most active speaker
        most_active = "unknown"
        max_time = 0
        for speaker_id, insights in individual_insights.items():
            speaking_time = insights.get('participation_metrics', {}).get('total_speaking_time', 0)
            if speaking_time > max_time:
                max_time = speaking_time
                most_active = speaker_id
        
        return f"""
Complete meeting analysis with LLM topic extraction finished successfully.

Using Llama 3.1 8B, we intelligently identified {len(topics)} main discussion topics with {num_speakers} active speakers participating.

The LLM was able to understand implicit topics without relying on keywords like "project" or "initiative".

Speaker Analysis Results:
- Most active participant: Speaker {most_active} with {max_time:.1f} seconds of speaking time
- All speakers showed distinct communication patterns and participation levels
- Sentiment analysis revealed varying emotional tones across participants

Key topics included: {', '.join([t.get('topic', 'Topic')[:25] for t in topics[:3]])}.

The comprehensive analysis used advanced LLM understanding to extract meaningful insights about both content and speaker behavior.
Processing completed in {processing_time:.1f} seconds.

All analysis files including LLM-extracted topics and speaker-specific insights have been saved and are ready for review.
"""

# Main function for compatibility
def main_LLM(meeting_starttime: str, 
             meeting_endtime: str, 
             transcript_path: str, 
             row_idx: int, 
             docx_path: str, 
             merged_output: List[str]) -> str:
    """Main LLM function with complete analysis including LLM topic extraction"""
    
    print(f"=== COMPLETE MAIN LLM FUNCTION WITH LLAMA 3.1 ===")
    print(f"Input: {len(merged_output)} segments")
    
    if not merged_output:
        return "Error: No transcript data"
    
    os.makedirs('processed', exist_ok=True)
    
    try:
        # Run complete async workflow with LLM
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        workflow = CompleteLLMWorkflow()
        result = loop.run_until_complete(
            workflow.run_complete_analysis(merged_output, meeting_starttime, meeting_endtime)
        )
        
        loop.close()
        return result
        
    except Exception as e:
        print(f"Error in complete main_LLM with Llama: {e}")
        import traceback
        traceback.print_exc()
        return f"Complete LLM processing failed: {str(e)}"

if __name__ == "__main__":
    # Test with sample data
    sample_data = [
        "0.0 --> 2.30 | A\nWe need to review the Alpha system timeline and make sure we're on track\n\n",
        "2.30 --> 5.15 | B\nI agree with the Alpha approach, but I have concerns about the budget allocation\n\n",
        "5.15 --> 8.45 | A\nLet's discuss the Beta feature requirements and see what we can optimize\n\n"
    ]
    
    result = main_LLM("14:00", "14:30", "test.txt", 0, "test.docx", sample_data)
    print("Result:", result)
