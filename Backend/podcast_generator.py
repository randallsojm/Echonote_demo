from typing import Dict, List
import re

class PodcastScriptGenerator:
    def __init__(self):
        self.transition_phrases = [
            "Moving on to our next topic",
            "Let's shift our focus to",
            "Another interesting point that came up",
            "Now, turning our attention to",
            "Speaking of which",
            "This brings us to",
            "On a related note"
        ]
        
        self.speaker_introductions = [
            "One participant mentioned",
            "Another speaker pointed out",
            "Someone in the discussion noted",
            "A participant emphasized",
            "One of the speakers highlighted"
        ]
    
    def generate_podcast_script(self, 
                              analysis_document: str,
                              topic_analysis: List[Dict],
                              speaker_analysis: Dict,
                              style: str = "conversational") -> str:
        """Generate a podcast script from the analysis"""
        
        script_sections = []
        
        # Introduction
        script_sections.append(self._generate_introduction(topic_analysis, speaker_analysis))
        
        # Main content sections
        script_sections.append(self._generate_topic_discussion(topic_analysis))
        script_sections.append(self._generate_speaker_insights_narration(speaker_analysis))
        script_sections.append(self._generate_key_takeaways(analysis_document))
        
        # Conclusion
        script_sections.append(self._generate_conclusion())
        
        # Combine all sections
        full_script = '\n\n'.join(script_sections)
        
        # Apply style formatting
        if style == "formal":
            full_script = self._apply_formal_style(full_script)
        else:
            full_script = self._apply_conversational_style(full_script)
        
        return full_script
    
    def _generate_introduction(self, topic_analysis: List[Dict], speaker_analysis: Dict) -> str:
        """Generate podcast introduction"""
        num_topics = len(topic_analysis)
        num_speakers = len(speaker_analysis.get('individual_insights', {}))
        
        intro = f"""[PODCAST INTRODUCTION]

Welcome to our meeting analysis podcast. Today, we're diving deep into a fascinating discussion that covered {num_topics} major topics with {num_speakers} active participants.

Using advanced AI analysis, we've extracted key insights, identified speaker perspectives, and uncovered the most important takeaways from this conversation. 

Let's start by exploring what made this discussion particularly interesting."""
        
        return intro
    
    def _generate_topic_discussion(self, topic_analysis: List[Dict]) -> str:
        """Generate topic-by-topic discussion for podcast"""
        sections = ["[MAIN TOPIC DISCUSSION]"]
        
        for i, topic_data in enumerate(topic_analysis):
            topic_name = topic_data.get('topic', f'Topic {i+1}')
            
            # Clean up topic name for speech
            clean_topic = re.sub(r'^(Project:|Discussion:)\s*', '', topic_name)
            
            if i == 0:
                sections.append(f"Let's begin with {clean_topic}.")
            else:
                transition = self.transition_phrases[i % len(self.transition_phrases)]
                sections.append(f"{transition} {clean_topic}.")
            
            # Add topic content
            if 'content' in topic_data:
                content = topic_data['content']
                
                # Extract key sentences for narration
                sentences = re.split(r'[.!?]+', content)
                key_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
                
                if key_sentences:
                    sections.append("Here's what was discussed:")
                    for sentence in key_sentences:
                        # Make it more speech-friendly
                        speech_sentence = self._make_speech_friendly(sentence)
                        sections.append(f"- {speech_sentence}")
            
            # Add speaker information
            if 'speakers' in topic_data:
                speakers = topic_data['speakers']
                if len(speakers) == 1:
                    sections.append(f"This point was primarily raised by Speaker {speakers[0]}.")
                else:
                    speaker_list = ', '.join([f"Speaker {s}" for s in speakers[:-1]])
                    sections.append(f"This was discussed by {speaker_list}, and Speaker {speakers[-1]}.")
            
            sections.append("")  # Add spacing
        
        return '\n'.join(sections)
    
    def _generate_speaker_insights_narration(self, speaker_analysis: Dict) -> str:
        """Generate speaker insights section for podcast"""
        sections = ["[SPEAKER INSIGHTS]"]
        
        sections.append("Now, let's look at how different participants contributed to this discussion.")
        
        individual_insights = speaker_analysis.get('individual_insights', {})
        
        for speaker_id, insights in individual_insights.items():
            sections.append(f"\nSpeaker {speaker_id} had some interesting characteristics:")
            
            # Participation style
            metrics = insights.get('participation_metrics', {})
            speaking_time = metrics.get('total_speaking_time', 0)
            segment_count = metrics.get('segment_count', 0)
            
            if speaking_time > 60:
                sections.append(f"They were quite active, speaking for over a minute total across {segment_count} contributions.")
            elif speaking_time > 30:
                sections.append(f"They participated moderately with {segment_count} contributions.")
            else:
                sections.append(f"They made {segment_count} focused contributions to the discussion.")
            
            # Sentiment and style
            sentiment = insights.get('sentiment_analysis', {})
            sentiment_trend = sentiment.get('sentiment_trend', 'neutral')
            
            if sentiment_trend == 'positive':
                sections.append("Their overall tone was positive and supportive.")
            elif sentiment_trend == 'negative':
                sections.append("They expressed some concerns or critical viewpoints.")
            else:
                sections.append("They maintained a balanced, neutral tone throughout.")
            
            # Key contributions
            contributions = insights.get('key_contributions', [])
            if contributions:
                sections.append("Their most significant contribution was:")
                # Take the first contribution and make it speech-friendly
                key_contribution = self._make_speech_friendly(contributions[0])
                sections.append(f'"{key_contribution}"')
        
        # Comparative insights
        comp_analysis = speaker_analysis.get('comparative_analysis', {})
        if comp_analysis and 'key_differences' in comp_analysis:
            sections.append("\nSome interesting dynamics emerged between the speakers:")
            for difference in comp_analysis['key_differences'][:3]:
                speech_difference = self._make_speech_friendly(difference)
                sections.append(f"- {speech_difference}")
        
        return '\n'.join(sections)
    
    def _generate_key_takeaways(self, analysis_document: str) -> str:
        """Generate key takeaways section"""
        sections = ["[KEY TAKEAWAYS]"]
        
        sections.append("So, what are the main takeaways from this discussion?")
        
        # Extract key findings from the document
        lines = analysis_document.split('\n')
        key_findings = []
        
        in_findings_section = False
        for line in lines:
            if 'KEY FINDINGS' in line.upper():
                in_findings_section = True
                continue
            elif line.startswith('##') and in_findings_section:
                break
            elif in_findings_section and line.strip().startswith('-'):
                finding = line.strip()[1:].strip()
                if len(finding) > 10:
                    key_findings.append(finding)
        
        if key_findings:
            sections.append("First, let me highlight the key decisions and findings:")
            for i, finding in enumerate(key_findings[:5], 1):
                speech_finding = self._make_speech_friendly(finding)
                sections.append(f"{i}. {speech_finding}")
        
        # Add general takeaways
        sections.append("\nOverall, this discussion demonstrates the value of structured dialogue and diverse perspectives in reaching meaningful conclusions.")
        
        return '\n'.join(sections)
    
    def _generate_conclusion(self) -> str:
        """Generate podcast conclusion"""
        conclusion = """[CONCLUSION]

That wraps up our analysis of this meeting discussion. We've seen how different speakers contributed unique perspectives, how topics evolved throughout the conversation, and what key decisions emerged.

This kind of AI-powered analysis helps us understand not just what was said, but how it was said, who said it, and what it all means in context.

Thank you for listening, and we hope this analysis provides valuable insights into effective communication and decision-making processes."""
        
        return conclusion
    
    def _make_speech_friendly(self, text: str) -> str:
        """Convert text to be more speech-friendly"""
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
        
        # Replace abbreviations
        replacements = {
            'e.g.': 'for example',
            'i.e.': 'that is',
            'etc.': 'and so on',
            '&': 'and',
            '%': 'percent'
        }
        
        for abbrev, replacement in replacements.items():
            text = text.replace(abbrev, replacement)
        
        # Ensure proper sentence structure
        text = text.strip()
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def _apply_conversational_style(self, script: str) -> str:
        """Apply conversational style to the script"""
        # Add conversational markers
        conversational_markers = [
            ("Let's", "Let's"),
            ("We can see", "We can see that"),
            ("It's interesting", "What's really interesting is"),
            ("This shows", "This clearly shows us"),
        ]
        
        for formal, conversational in conversational_markers:
            script = script.replace(formal, conversational)
        
        # Add pauses and emphasis markers
        script = re.sub(r'\. ([A-Z])', r'. [PAUSE] \1', script)
        script = re.sub(r'(important|key|significant|critical)', r'[EMPHASIS] \1 [/EMPHASIS]', script, flags=re.IGNORECASE)
        
        return script
    
    def _apply_formal_style(self, script: str) -> str:
        """Apply formal style to the script"""
        # Replace casual language
        formal_replacements = {
            "Let's": "We shall",
            "we're": "we are",
            "it's": "it is",
            "that's": "that is",
            "Here's": "Here is"
        }
        
        for casual, formal in formal_replacements.items():
            script = script.replace(casual, formal)
        
        return script
