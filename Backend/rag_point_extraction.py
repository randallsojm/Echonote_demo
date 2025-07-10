import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import re

class RAGPointExtractor:
    def __init__(self):
        print("Initializing RAG Point Extractor...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # MiniLM embedding dimension
        self.index = None
        self.text_chunks = []
        self.chunk_metadata = []
        print("RAG Point Extractor initialized")
        
    def build_faiss_index(self, merged_output: List[str]) -> None:
        """Build FAISS index from transcript chunks"""
        print("Building FAISS index from transcript...")
        
        # Process transcript chunks
        chunks = []
        metadata = []
        
        for i, segment in enumerate(merged_output):
            if not segment.strip():
                continue
                
            lines = segment.strip().split('\n')
            if len(lines) < 2:
                continue
                
            header = lines[0]
            content = '\n'.join(lines[1:]).strip()
            
            if not content:
                continue
            
            # Parse speaker and timestamp - NO 'Unknown' defaults
            speaker_match = re.search(r'\|\s*([A-Z])\s*$', header)
            time_match = re.search(r'(\d+\.\d+)\s*-->\s*(\d+\.\d+)', header)
            
            # Use actual speaker label or skip if can't parse
            if speaker_match:
                speaker = speaker_match.group(1)
            else:
                # Skip segments where we can't identify speaker rather than using 'Unknown'
                continue
                
            start_time = float(time_match.group(1)) if time_match else 0.0
            
            # Split content into meaningful chunks
            sentences = re.split(r'[.!?]+', content)
            for j, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if len(sentence) > 25:  # Only substantial sentences
                    chunks.append(sentence)
                    metadata.append({
                        'segment_id': i,
                        'sentence_id': j,
                        'speaker': speaker,  # This will always be A, B, C, etc.
                        'start_time': start_time,
                        'full_content': content,
                        'header': header
                    })
        
        if not chunks:
            print("No valid chunks found for FAISS index")
            return
        
        # Generate embeddings
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.model.encode(chunks, show_progress_bar=False)
        
        # Build FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        self.text_chunks = chunks
        self.chunk_metadata = metadata
        
        print(f"FAISS index built with {len(chunks)} chunks")
    
    def extract_points_for_topic(self, topic_data: Dict, top_k: int = 15) -> Dict:
        """Extract main points for a specific topic using RAG"""
        if self.index is None:
            print("FAISS index not built. Call build_faiss_index first.")
            return {}
        
        topic_text = topic_data['topic']
        print(f"Extracting points for topic: {topic_text}")
        
        # Create query embedding
        query_embedding = self.model.encode([topic_text], show_progress_bar=False)
        faiss.normalize_L2(query_embedding)
        
        # Search for relevant chunks
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Retrieve relevant chunks
        relevant_chunks = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.text_chunks) and score > 0.3:  # Quality threshold
                chunk_data = {
                    'text': self.text_chunks[idx],
                    'metadata': self.chunk_metadata[idx],
                    'relevance_score': float(score),
                    'rank': i + 1
                }
                relevant_chunks.append(chunk_data)
        
        # Summarize chunks into main points
        main_points = self._summarize_chunks_to_points(relevant_chunks, topic_text)
        
        return {
            'topic': topic_text,
            'main_points': main_points,
            'relevant_chunks': relevant_chunks,
            'speakers_involved': list(set([chunk['metadata']['speaker'] for chunk in relevant_chunks]))
        }
    
    def _summarize_chunks_to_points(self, chunks: List[Dict], topic: str) -> List[str]:
        """Summarize relevant chunks into concise main points"""
        if not chunks:
            return []
        
        # Group chunks by speaker for comprehensive analysis
        speaker_chunks = {}
        for chunk in chunks:
            speaker = chunk['metadata']['speaker']  # This will be A, B, C, etc.
            if speaker not in speaker_chunks:
                speaker_chunks[speaker] = []
            speaker_chunks[speaker].append(chunk['text'])
        
        main_points = []
        
        # Extract key themes from high-relevance chunks
        high_relevance_chunks = [c for c in chunks if c['relevance_score'] > 0.6]
        
        if high_relevance_chunks:
            # Combine high-relevance text for analysis
            combined_text = ' '.join([c['text'] for c in high_relevance_chunks[:8]])
            
            # Extract key points using comprehensive analysis
            sentences = re.split(r'[.!?]+', combined_text)
            key_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:
                    # Prioritize sentences with important indicators
                    importance_indicators = [
                        'important', 'critical', 'key', 'main', 'primary', 'essential',
                        'should', 'must', 'need', 'require', 'decision', 'conclude',
                        'agree', 'disagree', 'propose', 'suggest', 'recommend',
                        'problem', 'solution', 'issue', 'concern', 'risk'
                    ]
                    
                    sentence_score = sum(1 for indicator in importance_indicators if indicator in sentence.lower())
                    
                    if sentence_score > 0 or len(key_sentences) < 5:
                        key_sentences.append((sentence, sentence_score))
            
            # Sort by importance and take top sentences
            key_sentences.sort(key=lambda x: x[1], reverse=True)
            main_points.extend([sent[0] for sent in key_sentences[:8]])
        
        # Add speaker-specific insights if multiple speakers involved
        if len(speaker_chunks) > 1:
            for speaker, speaker_texts in speaker_chunks.items():
                if len(speaker_texts) >= 2:  # Speaker contributed significantly
                    combined_speaker_text = ' '.join(speaker_texts[:3])
                    if len(combined_speaker_text) > 50:
                        main_points.append(f"Speaker {speaker} perspective: {combined_speaker_text[:200]}...")
        
        # Remove duplicates and clean up
        unique_points = []
        seen = set()
        for point in main_points:
            point_key = point.lower()[:50]  # Use first 50 chars as key
            if point_key not in seen:
                unique_points.append(point)
                seen.add(point_key)
        
        return unique_points[:10]  # Return top 10 main points
