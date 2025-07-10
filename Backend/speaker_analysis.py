from typing import List, Dict, Set
import re
from collections import defaultdict, Counter

class SpeakerAnalyzer:
    def __init__(self):
        self.sentiment_positive = [
            'agree', 'good', 'excellent', 'great', 'positive', 'support',
            'like', 'love', 'appreciate', 'thank', 'wonderful', 'perfect'
        ]
        self.sentiment_negative = [
            'disagree', 'bad', 'terrible', 'negative', 'oppose', 'dislike',
            'hate', 'concern', 'worry', 'problem', 'issue', 'difficult'
        ]
        self.agreement_indicators = [
            'agree', 'exactly', 'right', 'correct', 'yes', 'absolutely',
            'definitely', 'support', 'endorse'
        ]
        self.disagreement_indicators = [
            'disagree', 'no', 'wrong', 'incorrect', 'oppose', 'against',
            'however', 'but', 'although', 'nevertheless'
        ]
    
    def analyze_speakers(self, merged_output: List[str]) -> Dict:
        """Analyze speaker-specific insights from diarised transcript"""
        speaker_data = defaultdict(list)
        speaker_stats = defaultdict(lambda: {
            'total_time': 0,
            'segment_count': 0,
            'word_count': 0,
            'topics_mentioned': set(),
            'sentiment_scores': [],
            'agreements': [],
            'disagreements': []
        })
        
        # Process each segment
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
            
            # Parse speaker and timestamp - NO 'Unknown' defaults
            speaker_match = re.search(r'\|\s*([A-Z])\s*$', header)
            time_match = re.search(r'(\d+\.\d+)\s*-->\s*(\d+\.\d+)', header)
            
            # Only process segments with valid speaker labels (A, B, C, etc.)
            if not speaker_match:
                print(f"Warning: Could not parse speaker from header '{header}', skipping segment")
                continue
                
            speaker = speaker_match.group(1)
            
            # Validate speaker is single uppercase letter
            if not (len(speaker) == 1 and speaker.isalpha() and speaker.isupper()):
                print(f"Warning: Invalid speaker format '{speaker}', skipping segment")
                continue
                
            start_time = float(time_match.group(1)) if time_match else 0.0
            end_time = float(time_match.group(2)) if time_match else 0.0
            
            # Store segment data
            segment_data = {
                'content': content,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'header': header
            }
            
            speaker_data[speaker].append(segment_data)
            
            # Update statistics
            stats = speaker_stats[speaker]
            stats['total_time'] += segment_data['duration']
            stats['segment_count'] += 1
            stats['word_count'] += len(content.split())
            
            # Analyze sentiment
            sentiment = self._analyze_sentiment(content)
            stats['sentiment_scores'].append(sentiment)
            
            # Detect agreements/disagreements
            agreements, disagreements = self._detect_agreements_disagreements(content)
            stats['agreements'].extend(agreements)
            stats['disagreements'].extend(disagreements)
            
            # Extract topics mentioned
            topics = self._extract_speaker_topics(content)
            stats['topics_mentioned'].update(topics)
        
        # Validate we have speakers to analyze
        if not speaker_data:
            print("Warning: No valid speakers found for analysis")
            return {
                'individual_insights': {},
                'comparative_analysis': {'note': 'No valid speakers found for analysis'},
                'speaker_statistics': {}
            }
        
        print(f"Analyzing {len(speaker_data)} speakers: {sorted(list(speaker_data.keys()))}")
        
        # Generate insights for each speaker
        speaker_insights = {}
        for speaker, segments in speaker_data.items():
            insights = self._generate_speaker_insights(speaker, segments, speaker_stats[speaker])
            speaker_insights[speaker] = insights
        
        # Generate comparative analysis
        comparative_analysis = self._generate_comparative_analysis(speaker_insights, speaker_stats)
        
        return {
            'individual_insights': speaker_insights,
            'comparative_analysis': comparative_analysis,
            'speaker_statistics': dict(speaker_stats)
        }
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis based on keyword matching"""
        text_lower = text.lower()
        positive_count = sum(1 for word in self.sentiment_positive if word in text_lower)
        negative_count = sum(1 for word in self.sentiment_negative if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Normalize by text length
        sentiment_score = (positive_count - negative_count) / max(total_words / 10, 1)
        return max(-1.0, min(1.0, sentiment_score))  # Clamp between -1 and 1
    
    def _detect_agreements_disagreements(self, text: str) -> tuple:
        """Detect agreement and disagreement expressions"""
        text_lower = text.lower()
        
        agreements = []
        disagreements = []
        
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if not sentence:
                continue
            
            # Check for agreement indicators
            for indicator in self.agreement_indicators:
                if indicator in sentence:
                    agreements.append(sentence[:100])
                    break
            
            # Check for disagreement indicators
            for indicator in self.disagreement_indicators:
                if indicator in sentence:
                    disagreements.append(sentence[:100])
                    break
        
        return agreements, disagreements
    
    def _extract_speaker_topics(self, text: str) -> Set[str]:
        """Extract topics mentioned by speaker"""
        topics = set()
        
        # Look for project mentions
        project_patterns = [
            r'project\s+([a-zA-Z][a-zA-Z0-9\s]{1,15})',
            r'([a-zA-Z][a-zA-Z0-9\s]{1,15})\s+project'
        ]
        
        for pattern in project_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_topic = re.sub(r'\s+', ' ', match.strip())
                if len(clean_topic) > 2:
                    topics.add(clean_topic.lower())
        
        return topics
    
    def _generate_speaker_insights(self, speaker: str, segments: List[Dict], stats: Dict) -> Dict:
        """Generate comprehensive insights for a specific speaker"""
        # Calculate averages
        avg_sentiment = sum(stats['sentiment_scores']) / len(stats['sentiment_scores']) if stats['sentiment_scores'] else 0
        avg_segment_duration = stats['total_time'] / stats['segment_count'] if stats['segment_count'] > 0 else 0
        
        # Identify key contributions
        longest_segments = sorted(segments, key=lambda x: x['duration'], reverse=True)[:3]
        key_contributions = [seg['content'][:200] + "..." for seg in longest_segments]
        
        # Communication style analysis
        communication_style = self._analyze_communication_style(segments)
        
        return {
            'speaker_id': speaker,
            'participation_metrics': {
                'total_speaking_time': round(stats['total_time'], 2),
                'segment_count': stats['segment_count'],
                'average_segment_duration': round(avg_segment_duration, 2),
                'total_words': stats['word_count']
            },
            'sentiment_analysis': {
                'average_sentiment': round(avg_sentiment, 3),
                'sentiment_trend': 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral'
            },
            'interaction_patterns': {
                'agreements_count': len(stats['agreements']),
                'disagreements_count': len(stats['disagreements']),
                'sample_agreements': stats['agreements'][:3],
                'sample_disagreements': stats['disagreements'][:3]
            },
            'topics_contributed': list(stats['topics_mentioned']),
            'key_contributions': key_contributions,
            'communication_style': communication_style
        }
    
    def _analyze_communication_style(self, segments: List[Dict]) -> Dict:
        """Analyze speaker's communication style"""
        all_text = ' '.join([seg['content'] for seg in segments])
        
        # Question frequency
        question_count = all_text.count('?')
        
        # Average sentence length
        sentences = re.split(r'[.!?]+', all_text)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        avg_sentence_length = sum(len(s.split()) for s in valid_sentences) / len(valid_sentences) if valid_sentences else 0
        
        # Formality indicators
        formal_words = ['therefore', 'however', 'furthermore', 'consequently', 'nevertheless']
        informal_words = ['yeah', 'okay', 'um', 'uh', 'like', 'you know']
        
        formal_count = sum(all_text.lower().count(word) for word in formal_words)
        informal_count = sum(all_text.lower().count(word) for word in informal_words)
        
        return {
            'questions_asked': question_count,
            'average_sentence_length': round(avg_sentence_length, 1),
            'formality_score': formal_count - informal_count,
            'communication_type': 'formal' if formal_count > informal_count else 'informal'
        }
    
    def _generate_comparative_analysis(self, speaker_insights: Dict, speaker_stats: Dict) -> Dict:
        """Generate comparative analysis between speakers"""
        if len(speaker_insights) < 2:
            return {'note': 'Comparative analysis requires at least 2 speakers'}
        
        # Speaking time comparison
        speaking_times = {speaker: insights['participation_metrics']['total_speaking_time'] 
                         for speaker, insights in speaker_insights.items()}
        
        most_active = max(speaking_times, key=speaking_times.get)
        least_active = min(speaking_times, key=speaking_times.get)
        
        # Sentiment comparison
        sentiments = {speaker: insights['sentiment_analysis']['average_sentiment'] 
                     for speaker, insights in speaker_insights.items()}
        
        most_positive = max(sentiments, key=sentiments.get)
        most_negative = min(sentiments, key=sentiments.get)
        
        # Agreement/disagreement patterns
        agreement_patterns = {}
        for speaker, insights in speaker_insights.items():
            agreements = insights['interaction_patterns']['agreements_count']
            disagreements = insights['interaction_patterns']['disagreements_count']
            agreement_patterns[speaker] = {
                'agreements': agreements,
                'disagreements': disagreements,
                'ratio': agreements / max(disagreements, 1)
            }
        
        return {
            'participation_analysis': {
                'most_active_speaker': most_active,
                'least_active_speaker': least_active,
                'speaking_time_distribution': speaking_times
            },
            'sentiment_comparison': {
                'most_positive_speaker': most_positive,
                'most_negative_speaker': most_negative,
                'sentiment_scores': sentiments
            },
            'interaction_dynamics': agreement_patterns,
            'key_differences': self._identify_key_differences(speaker_insights)
        }
    
    def _identify_key_differences(self, speaker_insights: Dict) -> List[str]:
        """Identify key differences between speakers"""
        differences = []
        
        speakers = list(speaker_insights.keys())
        if len(speakers) < 2:
            return differences
        
        # Compare communication styles
        for i, speaker_a in enumerate(speakers):
            for speaker_b in speakers[i+1:]:
                insights_a = speaker_insights[speaker_a]
                insights_b = speaker_insights[speaker_b]
                
                # Sentiment difference
                sent_a = insights_a['sentiment_analysis']['average_sentiment']
                sent_b = insights_b['sentiment_analysis']['average_sentiment']
                
                if abs(sent_a - sent_b) > 0.3:
                    if sent_a > sent_b:
                        differences.append(f"Speaker {speaker_a} was more positive than Speaker {speaker_b}")
                    else:
                        differences.append(f"Speaker {speaker_b} was more positive than Speaker {speaker_a}")
                
                # Participation difference
                time_a = insights_a['participation_metrics']['total_speaking_time']
                time_b = insights_b['participation_metrics']['total_speaking_time']
                
                if abs(time_a - time_b) > 30:  # 30 seconds difference
                    if time_a > time_b:
                        differences.append(f"Speaker {speaker_a} spoke significantly more than Speaker {speaker_b}")
                    else:
                        differences.append(f"Speaker {speaker_b} spoke significantly more than Speaker {speaker_a}")
        
        return differences
