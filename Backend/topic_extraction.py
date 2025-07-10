import requests
import json
import re
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class EnhancedLLMRAGSummarizer:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "llama3:8b"
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.knowledge_base = {}  # Store extracted insights for RAG
        
        print("Initializing Enhanced LLM+RAG Summarizer with Llama 3...")
        
        # Test connection
        if self._test_ollama_connection():
            print("Ollama connection successful")
        else:
            print("Ollama connection failed - make sure Ollama is running")
    
    def _test_ollama_connection(self) -> bool:
        """Test if Ollama is running and model is available"""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": "Hello",
                    "stream": False
                },
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Ollama connection error: {e}")
            return False
    
    def extract_topics_from_transcript(self, merged_output: List[str]) -> List[Dict]:
        """Extract topics using LLM analysis with enhanced contextual understanding"""
        print(f"Extracting topics from {len(merged_output)} segments using enhanced LLM+RAG analysis...")
        
        # Group segments into chunks for efficient processing
        chunks = self._group_segments_for_analysis(merged_output)
        all_topics = []
        
        # Build knowledge base from all chunks first
        self._build_knowledge_base(chunks)
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk['segments'])} segments)...")
            
            # Extract topics from this chunk using enhanced LLM+RAG
            chunk_topics = self._extract_topics_with_rag(chunk)
            all_topics.extend(chunk_topics)
            
            # Small delay to avoid overwhelming Ollama
            time.sleep(0.5)
        
        print(f"Extracted {len(all_topics)} topics using enhanced LLM+RAG analysis")
        return all_topics
    
    def _build_knowledge_base(self, chunks: List[Dict]) -> None:
        """Build a knowledge base from all conversation chunks for RAG"""
        print("Building knowledge base for RAG...")
        
        all_content = []
        chunk_metadata = []
        
        for i, chunk in enumerate(chunks):
            all_content.append(chunk['combined_text'])
            chunk_metadata.append({
                'chunk_id': i,
                'speakers': list(chunk['speakers']),
                'time_span': chunk['time_span'],
                'segment_count': len(chunk['segments'])
            })
        
        # Create TF-IDF vectors for semantic similarity
        try:
            self.content_vectors = self.vectorizer.fit_transform(all_content)
            self.knowledge_base = {
                'content': all_content,
                'metadata': chunk_metadata,
                'vectors': self.content_vectors
            }
            print(f"Knowledge base built with {len(all_content)} chunks")
        except Exception as e:
            print(f"Error building knowledge base: {e}")
            self.knowledge_base = {'content': all_content, 'metadata': chunk_metadata}
    
    def _get_relevant_context(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant context from knowledge base using semantic similarity"""
        if 'vectors' not in self.knowledge_base:
            return []
        
        try:
            # Transform query to vector space
            query_vector = self.vectorizer.transform([query_text])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.knowledge_base['vectors'])[0]
            
            # Get top-k most similar chunks
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            relevant_context = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    relevant_context.append({
                        'content': self.knowledge_base['content'][idx],
                        'metadata': self.knowledge_base['metadata'][idx],
                        'similarity': similarities[idx]
                    })
            
            return relevant_context
            
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []
    
    def _extract_topics_with_rag(self, chunk: Dict) -> List[Dict]:
        """Use LLM with RAG to extract topics and generate insightful analysis"""
        
        # Get relevant context from other parts of the conversation
        relevant_context = self._get_relevant_context(chunk['combined_text'], top_k=3)
        
        # Build context for RAG
        context_text = ""
        if relevant_context:
            context_text = "\n\nRELEVANT CONTEXT FROM OTHER PARTS OF THE CONVERSATION:\n"
            for i, ctx in enumerate(relevant_context):
                context_text += f"Context {i+1}: {ctx['content'][:300]}...\n"
        
        # --- Use user-provided topic extraction template for topic extraction ---
        topic_extraction_template = """
You are a helpful assistant that creates topics based on the context of a podcast transcript.
- Extract major ideas, themes, concepts, insights, comparisons, trends, viewpoints, and recurring points from the discussion.
- Each topic should be summarized with a brief, 1-sentence description of the idea or point.
- Use the exact language from the podcast for consistency.
- Do not include anything outside of the podcast discussion.
- Present each topic in the format: `Topic: Brief description`
"""

        # --- Use user-provided summary template for in-depth analysis ---
        summary_template = """
You will be given a podcast transcript containing multiple topics. Your task is to summarize key points for the topic the user chooses.

For each key point, provide a single, detailed paragraph (maximum 5 sentences) covering the following elements:
- Introduce the topic briefly based on the discussion.
- Identify challenges or problems raised, specifying who is facing them.
- Summarize the actions each speaker is taking and the progress of any related projects or initiatives.
- Outline any proposed ideas or solutions and the speaker suggesting them.
- Note any disagreements and the speakers involved.
- List follow-up actions and assign responsibility where applicable.

If any of the above details are missing, omit them rather than making assumptions.
Avoid preambles, introductory statements, and concluding remarks. Exclude irrelevant tangents or off-topic discussions.
Follow the format below:
### output format
```
1. <key point 1><One detailed paragraph about key point 1>

2. <key point 2><One detailed paragraph about key point 2>
```
-----------------
{context}
"""

        # --- Compose the prompt for the LLM ---
        prompt = f"""
{topic_extraction_template}

CURRENT CONVERSATION SEGMENT:
{chunk['combined_text']}

For each topic you identify, provide:
1. Topic name (be specific and descriptive)
2. Description: A brief, 1-sentence summary of the idea or point, using the exact language from the discussion where possible.
3. Analysis: For each topic, use the following template to generate an in-depth analysis:
{summary_template}
4. Disagreements: Only include **major disagreements**—specifically, cases where parties have clearly expressed opposition to each other's ideas or proposed clear alternatives. Do not include minor differences or simple clarifications. For each, provide a short summary (e.g., 'Speaker A said..., but Speaker B...').
5. Action items: ["specific action or decision, phrased in a similar tone to the above templates, and using the language of the discussion"]
6. Speakers: ["A", "B"]

IMPORTANT: You must return a valid JSON object with a 'topics' array, even if there is only one topic. Do not return plain text or a single object. If you find no topics, return an empty 'topics' array.

Format as JSON:
{{
  "topics": [
    {{
      "name": "specific descriptive topic name",
      "description": "A brief, 1-sentence summary of the idea or point, using the exact language from the discussion where possible.",
      "analysis": "A detailed, paragraph-style synthesis of the discussion, actions, and follow-ups for this topic.",
      "disagreements": ["Only major disagreements, e.g., 'Speaker A said..., but Speaker B...'"],
      "action_items": ["specific action or decision, phrased in a similar tone to the above templates, and using the language of the discussion"],
      "speakers": ["A", "B"]
    }}
  ]
}}
"""

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 2000  # Allow longer responses for detailed analysis
                    }
                },
                timeout=90
            )
            
            if response.status_code != 200:
                print(f"LLM request failed: {response.status_code}")
                return []
            
            llm_response = response.json().get('response', '')
            
            # Parse enhanced LLM response
            topics = self._parse_enhanced_llm_response(llm_response, chunk)
            
            # Further enhance with cross-topic analysis
            enhanced_topics = self._enhance_with_cross_topic_analysis(topics)
            
            return enhanced_topics
            
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return []
    
    def _parse_enhanced_llm_response(self, llm_response: str, chunk: Dict) -> List[Dict]:
        """Parse enhanced LLM response with detailed insights"""
        topics = []
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                if 'topics' in parsed and isinstance(parsed['topics'], list):
                    for topic_data in parsed['topics']:
                        topic = {
                            'topic': topic_data.get('name', 'Unknown Topic'),
                            'category': topic_data.get('category', 'General'),
                            'description': topic_data.get('description', ''),
                            'analysis': topic_data.get('analysis', ' '.join(topic_data.get('key_insights', []))),
                            'key_insights': topic_data.get('key_insights', []),
                            'action_items': topic_data.get('action_items', []),
                            'stakeholder_concerns': topic_data.get('stakeholder_concerns', []),
                            'relationships': topic_data.get('relationships', []),
                            'risks_challenges': topic_data.get('risks_challenges', []),
                            'next_steps': topic_data.get('next_steps', []),
                            'speakers': topic_data.get('speakers', list(chunk['speakers'])),
                            'confidence': topic_data.get('confidence', 0.8),
                            'sentiment': topic_data.get('sentiment', 'neutral'),
                            'urgency': topic_data.get('urgency', 'medium'),
                            'content': chunk['combined_text'],
                            'start_time': chunk['time_span']['start'],
                            'end_time': chunk['time_span']['end'],
                            'segments': chunk['segments']
                        }
                        topics.append(topic)
                elif 'topics' in parsed and isinstance(parsed['topics'], dict):
                    topic_data = parsed['topics']
                    topic = {
                        'topic': topic_data.get('name', 'Unknown Topic'),
                        'category': topic_data.get('category', 'General'),
                        'description': topic_data.get('description', ''),
                        'analysis': topic_data.get('analysis', ' '.join(topic_data.get('key_insights', []))),
                        'key_insights': topic_data.get('key_insights', []),
                        'action_items': topic_data.get('action_items', []),
                        'stakeholder_concerns': topic_data.get('stakeholder_concerns', []),
                        'relationships': topic_data.get('relationships', []),
                        'risks_challenges': topic_data.get('risks_challenges', []),
                        'next_steps': topic_data.get('next_steps', []),
                        'speakers': topic_data.get('speakers', list(chunk['speakers'])),
                        'confidence': topic_data.get('confidence', 0.8),
                        'sentiment': topic_data.get('sentiment', 'neutral'),
                        'urgency': topic_data.get('urgency', 'medium'),
                        'content': chunk['combined_text'],
                        'start_time': chunk['time_span']['start'],
                        'end_time': chunk['time_span']['end'],
                        'segments': chunk['segments']
                    }
                    topics.append(topic)
                else:
                    print("[LLM DEBUG] No 'topics' array found in parsed JSON. Parsed object:", parsed)
        except Exception as e:
            print("[LLM DEBUG] JSON parsing failed, attempting enhanced text extraction...", e)
            topics = self._extract_enhanced_topics_from_text(llm_response, chunk)
        if not topics:
            print("[LLM DEBUG] No topics parsed from LLM response. Raw response:\n", llm_response)
        return topics
    
    def _extract_enhanced_topics_from_text(self, response: str, chunk: Dict) -> List[Dict]:
        """Fallback: extract enhanced topics from plain text LLM response"""
        topics = []
        # Look for topic patterns with enhanced information
        sections = re.split(r'\n\s*\n', response)
        for section in sections:
            if not section.strip():
                continue
            # Look for topic name
            topic_match = re.search(r'(?:topic|name):\s*([^\n]+)', section, re.IGNORECASE)
            if topic_match:
                topic_name = topic_match.group(1).strip()
                # Extract description
                desc_match = re.search(r'(?:description):\s*([^\n]+)', section, re.IGNORECASE)
                description = desc_match.group(1).strip() if desc_match else ''
                # Extract analysis
                analysis_match = re.search(r'(?:analysis):\s*([^\n]+)', section, re.IGNORECASE)
                analysis = analysis_match.group(1).strip() if analysis_match else ''
                # Extract insights
                insights = re.findall(r'(?:insight|analysis):\s*([^\n]+)', section, re.IGNORECASE)
                actions = re.findall(r'(?:action|decision):\s*([^\n]+)', section, re.IGNORECASE)
                topic = {
                    'topic': topic_name,
                    'category': 'General',
                    'description': description,
                    'analysis': analysis or ' '.join(insights),
                    'key_insights': insights,
                    'action_items': actions,
                    'stakeholder_concerns': [],
                    'relationships': [],
                    'risks_challenges': [],
                    'next_steps': [],
                    'speakers': list(chunk['speakers']),
                    'confidence': 0.7,
                    'sentiment': 'neutral',
                    'urgency': 'medium',
                    'content': chunk['combined_text'],
                    'start_time': chunk['time_span']['start'],
                    'end_time': chunk['time_span']['end'],
                    'segments': chunk['segments']
                }
                topics.append(topic)
        if not topics:
            print("[LLM DEBUG] Fallback parser found no topics in sectioned text. Raw response:\n", response)
        return topics
    
    def _enhance_with_cross_topic_analysis(self, topics: List[Dict]) -> List[Dict]:
        """Enhance topics with cross-topic relationship analysis"""
        if len(topics) <= 1:
            return topics
        
        # Analyze relationships between topics
        for i, topic in enumerate(topics):
            related_topics = []
            
            for j, other_topic in enumerate(topics):
                if i != j:
                    # Check for relationships based on content similarity
                    if self._topics_are_related(topic, other_topic):
                        related_topics.append(other_topic['topic'])
            
            # Update relationships
            if related_topics:
                existing_relationships = topic.get('relationships', [])
                topic['relationships'] = list(set(existing_relationships + related_topics))
        
        return topics
    
    def _topics_are_related(self, topic1: Dict, topic2: Dict) -> bool:
        """Determine if two topics are related based on content and metadata"""
        # Check for speaker overlap
        speakers1 = set(topic1.get('speakers', []))
        speakers2 = set(topic2.get('speakers', []))
        
        if speakers1 & speakers2:  # Common speakers
            # Check for content similarity
            content1 = topic1.get('content', '')
            content2 = topic2.get('content', '')
            
            # Simple keyword overlap check
            words1 = set(content1.lower().split())
            words2 = set(content2.lower().split())
            
            common_words = words1 & words2
            if len(common_words) > 5:  # Threshold for relatedness
                return True
        
        return False
    
    def generate_comprehensive_summary(self, topics: List[Dict]) -> Dict:
        """Generate a comprehensive meeting summary using LLM+RAG"""
        print("Generating comprehensive meeting summary...")
        
        # Organize topics by category
        categorized_topics = defaultdict(list)
        for topic in topics:
            category = topic.get('category', 'General')
            categorized_topics[category].append(topic)
        
        # Generate summary for each category
        category_summaries = {}
        for category, category_topics in categorized_topics.items():
            category_summaries[category] = self._generate_category_summary(category, category_topics)
        
        # Generate overall meeting insights
        overall_insights = self._generate_overall_insights(topics)
        
        return {
            'overall_insights': overall_insights,
            'category_summaries': category_summaries,
            'topic_count': len(topics),
            'categories': list(categorized_topics.keys()),
            'high_priority_items': self._extract_high_priority_items(topics),
            'action_items': self._consolidate_action_items(topics),
            'risk_factors': self._consolidate_risks(topics),
            'next_steps': self._consolidate_next_steps(topics)
        }
    
    def _generate_category_summary(self, category: str, topics: List[Dict]) -> Dict:
        """Generate summary for a specific category of topics"""
        topics_text = ""
        for topic in topics:
            topics_text += f"Topic: {topic['topic']}\n"
            topics_text += f"Insights: {'; '.join(topic.get('key_insights', []))}\n"
            topics_text += f"Actions: {'; '.join(topic.get('action_items', []))}\n\n"
        
        prompt = f"""Analyze these {category} topics from a meeting and provide a comprehensive summary:

{topics_text}

Provide:
1. Key themes and patterns
2. Critical decisions made
3. Outstanding issues or concerns
4. Resource implications
5. Strategic impact
6. Recommended priorities

Format as JSON:
{{
  "key_themes": ["theme1", "theme2"],
  "critical_decisions": ["decision1", "decision2"],
  "outstanding_issues": ["issue1", "issue2"],
  "resource_implications": ["implication1", "implication2"],
  "strategic_impact": "overall impact assessment",
  "recommended_priorities": ["priority1", "priority2"]
}}"""

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                },
                timeout=60
            )
            
            if response.status_code == 200:
                llm_response = response.json().get('response', '')
                return self._parse_category_summary(llm_response)
            
        except Exception as e:
            print(f"Error generating category summary: {e}")
        
        return {"summary": f"Summary for {category} topics", "topics": topics}
    
    def _generate_overall_insights(self, topics: List[Dict]) -> Dict:
        """Generate overall meeting insights using LLM analysis"""
        all_insights = []
        all_concerns = []
        
        for topic in topics:
            all_insights.extend(topic.get('key_insights', []))
            all_concerns.extend(topic.get('stakeholder_concerns', []))
        
        insights_text = '\n'.join(all_insights)
        concerns_text = '\n'.join(all_concerns)
        
        prompt = f"""Based on these meeting insights and concerns, provide overall meeting analysis:

INSIGHTS:
{insights_text}

CONCERNS:
{concerns_text}

Provide:
1. Meeting effectiveness assessment
2. Key outcomes and achievements
3. Major risks or roadblocks
4. Team alignment and engagement
5. Strategic implications
6. Recommended follow-up actions

Format as JSON with detailed analysis."""

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                },
                timeout=60
            )
            
            if response.status_code == 200:
                llm_response = response.json().get('response', '')
                return self._parse_overall_insights(llm_response)
            
        except Exception as e:
            print(f"Error generating overall insights: {e}")
        
        return {"effectiveness": "moderate", "outcomes": [], "risks": []}
    
    def _parse_category_summary(self, llm_response: str) -> Dict:
        """Parse category summary from LLM response"""
        try:
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass
        
        return {"summary": llm_response[:500]}
    
    def _parse_overall_insights(self, llm_response: str) -> Dict:
        """Parse overall insights from LLM response"""
        try:
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass
        
        return {"analysis": llm_response[:500]}
    
    def _extract_high_priority_items(self, topics: List[Dict]) -> List[Dict]:
        """Extract high priority items based on urgency and risks"""
        high_priority = []
        
        for topic in topics:
            urgency = topic.get('urgency', 'medium')
            risks = topic.get('risks_challenges', [])
            
            if urgency == 'high' or len(risks) > 0:
                high_priority.append({
                    'topic': topic['topic'],
                    'urgency': urgency,
                    'risks': risks,
                    'actions': topic.get('action_items', [])
                })
        
        return high_priority
    
    def _consolidate_action_items(self, topics: List[Dict]) -> List[str]:
        """Consolidate all action items across topics"""
        all_actions = []
        
        for topic in topics:
            actions = topic.get('action_items', [])
            for action in actions:
                all_actions.append(f"[{topic['topic']}] {action}")
        
        return all_actions
    
    def _consolidate_risks(self, topics: List[Dict]) -> List[str]:
        """Consolidate all risks across topics"""
        all_risks = []
        
        for topic in topics:
            risks = topic.get('risks_challenges', [])
            for risk in risks:
                all_risks.append(f"[{topic['topic']}] {risk}")
        
        return all_risks
    
    def _consolidate_next_steps(self, topics: List[Dict]) -> List[str]:
        """Consolidate all next steps across topics"""
        all_steps = []
        
        for topic in topics:
            steps = topic.get('next_steps', [])
            for step in steps:
                all_steps.append(f"[{topic['topic']}] {step}")
        
        return all_steps
    
    # Keep the original methods for backward compatibility
    def _group_segments_for_analysis(self, merged_output: List[str]) -> List[Dict]:
        """Group segments into manageable chunks for LLM processing"""
        chunks = []
        current_chunk = {
            'segments': [],
            'combined_text': '',
            'speakers': set(),
            'time_span': {'start': float('inf'), 'end': 0}
        }
        
        for segment in merged_output:
            if not segment.strip():
                continue
                
            lines = segment.strip().split('\n')
            if len(lines) < 2:
                continue
                
            header = lines[0]
            content = '\n'.join(lines[1:]).strip()
            
            if not content:
                continue
            
            # Parse speaker and timestamp
            speaker_match = re.search(r'\|\s*([A-Z])\s*$', header)
            time_match = re.search(r'(\d+\.\d+)\s*-->\s*(\d+\.\d+)', header)
            
            if not speaker_match:
                continue
                
            speaker = speaker_match.group(1)
            start_time = float(time_match.group(1)) if time_match else 0.0
            end_time = float(time_match.group(2)) if time_match else 0.0
            
            # Add to current chunk
            current_chunk['segments'].append({
                'speaker': speaker,
                'content': content,
                'start_time': start_time,
                'end_time': end_time,
                'header': header,
                'segment': segment
            })
            
            current_chunk['combined_text'] += f"Speaker {speaker}: {content}\n"
            current_chunk['speakers'].add(speaker)
            current_chunk['time_span']['start'] = min(current_chunk['time_span']['start'], start_time)
            current_chunk['time_span']['end'] = max(current_chunk['time_span']['end'], end_time)
            
            # If chunk gets too large, start a new one
            if len(current_chunk['combined_text']) > 2000:
                chunks.append(current_chunk)
                current_chunk = {
                    'segments': [],
                    'combined_text': '',
                    'speakers': set(),
                    'time_span': {'start': float('inf'), 'end': 0}
                }
        
        # Add final chunk if it has content
        if current_chunk['segments']:
            chunks.append(current_chunk)
        
        return chunks
    
    def deduplicate_topics(self, topics: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
        """Deduplicate topics using enhanced LLM-based similarity analysis"""
        if len(topics) <= 1:
            return topics
        
        print(f"Deduplicating {len(topics)} topics using enhanced LLM analysis...")
        
        # Use more sophisticated deduplication considering insights and relationships
        deduplicated = self._advanced_topic_deduplication(topics)
        
        print(f"After enhanced deduplication: {len(deduplicated)} topics")
        return deduplicated
    
    def _advanced_topic_deduplication(self, topics: List[Dict]) -> List[Dict]:
        """Advanced deduplication considering topic insights and relationships"""
        # Group by category first
        category_groups = defaultdict(list)
        for topic in topics:
            category = topic.get('category', 'General')
            category_groups[category].append(topic)
        
        deduplicated = []
        for category, group in category_groups.items():
            if len(group) == 1:
                deduplicated.extend(group)
            else:
                # Use semantic similarity for deduplication
                merged_group = self._semantic_topic_merge(group)
                deduplicated.extend(merged_group)
        
        return deduplicated
    
    def _semantic_topic_merge(self, topics: List[Dict]) -> List[Dict]:
        """Merge topics using semantic similarity of their content and insights"""
        if len(topics) <= 1:
            return topics
        
        # Create feature vectors from topic content and insights
        topic_texts = []
        for topic in topics:
            text = topic['topic'] + ' ' + ' '.join(topic.get('key_insights', []))
            topic_texts.append(text)
        
        try:
            # Calculate similarity matrix
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            vectors = vectorizer.fit_transform(topic_texts)
            similarity_matrix = cosine_similarity(vectors)
            
            # Group similar topics
            merged_topics = []
            processed = set()
            
            for i, topic in enumerate(topics):
                if i in processed:
                    continue
                
                # Find similar topics
                similar_indices = [i]
                for j in range(i + 1, len(topics)):
                    if j not in processed and similarity_matrix[i][j] > 0.7:
                        similar_indices.append(j)
                        processed.add(j)
                
                processed.add(i)
                
                if len(similar_indices) == 1:
                    merged_topics.append(topic)
                else:
                    # Merge similar topics
                    topics_to_merge = [topics[idx] for idx in similar_indices]
                    merged_topic = self._merge_enhanced_topics(topics_to_merge)
                    merged_topics.append(merged_topic)
            
            return merged_topics
            
        except Exception as e:
            print(f"Error in semantic merging: {e}")
            return topics
    
    def _merge_enhanced_topics(self, topics_to_merge: List[Dict]) -> Dict:
        """Merge multiple enhanced topics into one comprehensive topic"""
        if not topics_to_merge:
            return {}
        
        # Combine all enhanced data
        all_insights = []
        all_actions = []
        all_concerns = []
        all_relationships = []
        all_risks = []
        all_next_steps = []
        all_speakers = set()
        all_segments = []
        
        min_start = float('inf')
        max_end = 0
        best_topic = topics_to_merge[0]  # Topic with highest confidence
        
        for topic in topics_to_merge:
            all_insights.extend(topic.get('key_insights', []))
            all_actions.extend(topic.get('action_items', []))
            all_concerns.extend(topic.get('stakeholder_concerns', []))
            all_relationships.extend(topic.get('relationships', []))
            all_risks.extend(topic.get('risks_challenges', []))
            all_next_steps.extend(topic.get('next_steps', []))
            all_speakers.update(topic.get('speakers', []))
            all_segments.extend(topic.get('segments', []))
            
            min_start = min(min_start, topic.get('start_time', 0))
            max_end = max(max_end, topic.get('end_time', 0))
            
            if topic.get('confidence', 0) > best_topic.get('confidence', 0):
                best_topic = topic
        
        # Remove duplicates
        all_insights = list(set(all_insights))
        all_actions = list(set(all_actions))
        all_concerns = list(set(all_concerns))
        all_relationships = list(set(all_relationships))
        all_risks = list(set(all_risks))
        all_next_steps = list(set(all_next_steps))
        
        return {
            'topic': best_topic['topic'],
            'category': best_topic.get('category', 'General'),
            'key_insights': all_insights,
            'action_items': all_actions,
            'stakeholder_concerns': all_concerns,
            'relationships': all_relationships,
            'risks_challenges': all_risks,
            'next_steps': all_next_steps,
            'speakers': list(all_speakers),
            'confidence': max(t.get('confidence', 0.8) for t in topics_to_merge),
            'sentiment': best_topic.get('sentiment', 'neutral'),
            'urgency': max([t.get('urgency', 'low') for t in topics_to_merge], 
                          key=lambda x: {'low': 0, 'medium': 1, 'high': 2}.get(x, 0)),
            'content': best_topic.get('content', ''),
            'start_time': min_start,
            'end_time': max_end,
            'segments': all_segments,
            'merged_count': len(topics_to_merge)
        }