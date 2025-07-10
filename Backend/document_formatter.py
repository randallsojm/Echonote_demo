from typing import Dict, List
from datetime import datetime
import json

class DocumentFormatter:
    def __init__(self):
        self.template_sections = [
            'executive_summary',
            'topic_analysis',
            'speaker_insights',
            'key_decisions',
            'action_items',
            'appendix'
        ]
    
    def format_comprehensive_analysis(self, 
                                   topic_analysis: List[Dict],
                                   speaker_analysis: Dict,
                                   rag_results: List[Dict],
                                   meeting_metadata: Dict = None,
                                   comprehensive_summary: Dict = None) -> str:
        """Format comprehensive analysis into a structured Markdown document"""
        document = []
        # Header (no YAML frontmatter)
        document.append(self._generate_header(meeting_metadata))
        # 1. Summary of meeting
        document.append(self._generate_meeting_summary_section(topic_analysis))
        # 2. Meeting Details (in-depth analysis and follow-up)
        document.append(self._generate_meeting_details_section(topic_analysis))
        # 3. Follow-up Actions
        document.append(self._generate_followup_actions_section(topic_analysis))
        # 4. Speaker Analysis
        document.append(self._generate_speaker_insights_section(speaker_analysis))
        # 5. Appendix (optional, keep if you want detailed stats)
        # document.append(self._generate_appendix(speaker_analysis))
        return '\n\n---\n\n'.join(document)
    
    def _generate_header(self, metadata: Dict = None) -> str:
        """Generate document header in Markdown"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"""# 📝 Meeting Analysis Report\n\n**Generated on:** {current_time}  \
**Analysis Type:** Comprehensive Speaker-Diarised Transcript Analysis\n"""
        if metadata:
            header += f"""
**Meeting Details:**
| Start Time | End Time | Duration | Participants |
|---|---|---|---|
| {metadata.get('start_time', 'N/A')} | {metadata.get('end_time', 'N/A')} | {metadata.get('duration', 'N/A')} | {metadata.get('participants', 'N/A')} |
"""
        return header
    
    def _generate_executive_summary(self, topic_analysis: List[Dict], speaker_analysis: Dict) -> str:
        """Generate executive summary section with a summary table"""
        summary = ["## Executive Summary 🗒️"]
        total_topics = len(topic_analysis)
        total_speakers = len(speaker_analysis.get('individual_insights', {}))
        summary.append(f"\n**Topics Discussed:** {total_topics}  \\n**Active Speakers:** {total_speakers}  \\n**Analysis Method:** RAG-based semantic analysis with speaker diarisation\n")
        # Speaker summary table
        speaker_rows = []
        for speaker_id, insights in speaker_analysis.get('individual_insights', {}).items():
            metrics = insights.get('participation_metrics', {})
            sentiment = insights.get('sentiment_analysis', {})
            speaker_rows.append(f"| {speaker_id} | {metrics.get('total_speaking_time', 0):.1f} | {metrics.get('segment_count', 0)} | {sentiment.get('average_sentiment', 0):.3f} |")
        if speaker_rows:
            summary.append("\n**Speaker Participation:**\n| Speaker | Speaking Time (s) | Contributions | Avg Sentiment |\n|---|---|---|---|\n" + '\n'.join(speaker_rows))
        # Key highlights
        comp_analysis = speaker_analysis.get('comparative_analysis', {})
        if comp_analysis:
            most_active = comp_analysis.get('participation_analysis', {}).get('most_active_speaker', 'N/A')
            most_positive = comp_analysis.get('sentiment_comparison', {}).get('most_positive_speaker', 'N/A')
            summary.append(f"\n**Key Highlights:**\n- **Most Active Participant:** Speaker {most_active}\n- **Most Positive Contributor:** Speaker {most_positive}\n- **Primary Discussion Areas:** {', '.join([t.get('topic', 'Unknown')[:50] for t in topic_analysis[:3]])}\n")
        return '\n'.join(summary)
    
    def _generate_meeting_summary_section(self, topic_analysis: List[Dict]) -> str:
        section = ["## Summary of meeting:"]
        for i, topic_data in enumerate(topic_analysis, 1):
            topic_name = topic_data.get('topic', f'Topic {i}')
            description = topic_data.get('description', '').strip()
            section.append(f"**{i}. ITEM {i} - {topic_name}**")
            if description:
                section.append(f"*{description}*\n")
        return '\n'.join(section)
    
    def _generate_meeting_details_section(self, topic_analysis: List[Dict]) -> str:
        section = ["## Meeting Details:"]
        for i, topic_data in enumerate(topic_analysis, 1):
            topic_name = topic_data.get('topic', f'Topic {i}')
            analysis = topic_data.get('analysis', '').strip()
            actions = topic_data.get('action_items', [])
            section.append(f"**{i}. ITEM {i} - {topic_name}**")
            if analysis:
                section.append(f"{analysis}\n")
            if actions:
                section.append(f"**Follow up actions required:**")
                for act in actions:
                    section.append(f"- {act}")
            section.append("\n---\n")
        return '\n'.join(section)
    
    def _generate_followup_actions_section(self, topic_analysis: List[Dict]) -> str:
        section = ["## Follow-up Actions:"]
        all_actions = []
        for topic_data in topic_analysis:
            topic_name = topic_data.get('topic', 'Unknown Topic')
            actions = topic_data.get('action_items', [])
            for act in actions:
                all_actions.append((topic_name, act))
        if all_actions:
            for i, (topic, act) in enumerate(all_actions, 1):
                section.append(f"{i}. [{topic}] {act}")
        else:
            section.append("*No follow-up actions identified.*")
        return '\n'.join(section)
    
    def _generate_speaker_insights_section(self, speaker_analysis: Dict) -> str:
        section = ["## Speaker Analysis:"]
        section.append("*Sentiment score reflects the overall emotional tone of the speaker's contributions, with positive values indicating positive sentiment, negative values indicating negative sentiment, and values near zero indicating neutrality.*\n")
        individual_insights = speaker_analysis.get('individual_insights', {})
        for speaker_id, insights in individual_insights.items():
            section.append(f"### Speaker {speaker_id}")
            metrics = insights.get('participation_metrics', {})
            sentiment = insights.get('sentiment_analysis', {})
            section.append(f"- **Total Speaking Time:** {metrics.get('total_speaking_time', 0):.1f} seconds")
            section.append(f"- **Total Words:** {metrics.get('total_words', 0)}")
            section.append(f"- **Sentiment Score:** {sentiment.get('average_sentiment', 0):.3f}")
            # Key Contributions
            contributions = insights.get('key_contributions', [])
            if contributions:
                section.append("\n**Key Contributions:**")
                for j, contribution in enumerate(contributions[:3], 1):
                    section.append(f"- {contribution}")
            section.append("\n---\n")
        return '\n'.join(section)
    
    def _generate_key_findings_section(self, topic_analysis: List[Dict], speaker_analysis: Dict) -> str:
        """Generate key findings and decisions section"""
        section = ["## Key Findings & Decisions 📌"]
        decisions = []
        for topic_data in topic_analysis:
            content = topic_data.get('content', '')
            if any(keyword in content.lower() for keyword in ['decide', 'decision', 'agree', 'conclude']):
                decisions.append({
                    'topic': topic_data.get('topic', 'Unknown'),
                    'content': content[:200] + "..."
                })
        if decisions:
            section.append("\n### Identified Decisions")
            for i, decision in enumerate(decisions, 1):
                section.append(f"- **{decision['topic']}**: {decision['content']}")
        disagreements = []
        individual_insights = speaker_analysis.get('individual_insights', {})
        for speaker_id, insights in individual_insights.items():
            speaker_disagreements = insights.get('interaction_patterns', {}).get('sample_disagreements', [])
            for disagreement in speaker_disagreements:
                disagreements.append(f"Speaker {speaker_id}: {disagreement}")
        if disagreements:
            section.append("\n### Areas of Disagreement")
            for i, disagreement in enumerate(disagreements[:5], 1):
                section.append(f"- {disagreement}")
        return '\n'.join(section)
    
    def _generate_action_items_section(self, rag_results: List[Dict]) -> str:
        """Generate action items section"""
        section = ["## Action Items & Next Steps ✅"]
        action_items = []
        for result in rag_results:
            main_points = result.get('main_points', [])
            for point in main_points:
                if any(keyword in point.lower() for keyword in ['should', 'need', 'must', 'will', 'action', 'next']):
                    action_items.append(point)
        if action_items:
            section.append("\n### Identified Action Items")
            for i, item in enumerate(action_items[:10], 1):
                section.append(f"- {item}")
        else:
            section.append("\n*No specific action items identified from the discussion.*")
        section.append("\n### Recommendations\n- Follow up on key decisions made during the meeting\n- Address any unresolved disagreements in future discussions\n- Ensure all participants are aligned on next steps")
        return '\n'.join(section)
    
    def _generate_appendix(self, speaker_analysis: Dict) -> str:
        """Generate appendix with detailed statistics as a Markdown table"""
        section = ["## Appendix 📎"]
        section.append("\n### Detailed Speaker Statistics")
        individual_insights = speaker_analysis.get('individual_insights', {})
        if individual_insights:
            section.append("| Speaker | Questions Asked | Avg Sentence Length | Communication Type | Formality Score |")
            section.append("|---|---|---|---|---|")
            for speaker_id, insights in individual_insights.items():
                comm_style = insights.get('communication_style', {})
                section.append(f"| {speaker_id} | {comm_style.get('questions_asked', 0)} | {comm_style.get('average_sentence_length', 0)} | {comm_style.get('communication_type', 'unknown')} | {comm_style.get('formality_score', 0)} |")
        comp_analysis = speaker_analysis.get('comparative_analysis', {})
        if comp_analysis:
            section.append("\n### Comparative Statistics")
            participation = comp_analysis.get('participation_analysis', {})
            if participation:
                section.append(f"- Most Active: Speaker {participation.get('most_active_speaker', 'N/A')}")
                section.append(f"- Least Active: Speaker {participation.get('least_active_speaker', 'N/A')}")
            sentiment_comp = comp_analysis.get('sentiment_comparison', {})
            if sentiment_comp:
                section.append(f"- Most Positive: Speaker {sentiment_comp.get('most_positive_speaker', 'N/A')}")
                section.append(f"- Most Negative: Speaker {sentiment_comp.get('most_negative_speaker', 'N/A')}")
        return '\n'.join(section)

    def _generate_comprehensive_summary_section(self, summary: Dict) -> str:
        """Generate a section summarizing insights from comprehensive_summary.json"""
        section = ["## LLM+RAG Comprehensive Summary 🧠"]
        # Overall Insights
        overall = summary.get('overall_insights', {})
        if overall:
            section.append("### Overall Meeting Insights")
            if isinstance(overall, dict) and 'analysis' in overall:
                section.append(f"{overall['analysis']}")
            else:
                section.append(f"{overall}")
        # Category Summaries
        categories = summary.get('category_summaries', {})
        if categories:
            section.append("\n### Category Summaries")
            for cat, cat_data in categories.items():
                section.append(f"\n#### {cat}")
                if 'key_themes' in cat_data:
                    section.append("- **Key Themes:** " + ", ".join(cat_data['key_themes']))
                if 'critical_decisions' in cat_data:
                    section.append("- **Critical Decisions:** " + ", ".join(cat_data['critical_decisions']))
                if 'outstanding_issues' in cat_data:
                    section.append("- **Outstanding Issues:** " + ", ".join(cat_data['outstanding_issues']))
                if 'resource_implications' in cat_data:
                    section.append("- **Resource Implications:** " + ", ".join(cat_data['resource_implications']))
                if 'strategic_impact' in cat_data:
                    section.append(f"- **Strategic Impact:** {cat_data['strategic_impact']}")
                if 'recommended_priorities' in cat_data:
                    section.append("- **Recommended Priorities:** " + ", ".join(cat_data['recommended_priorities']))
        # High Priority Items
        high_priority = summary.get('high_priority_items', [])
        if high_priority:
            section.append("\n### High Priority Topics & Risks")
            for item in high_priority:
                section.append(f"- **{item.get('topic','')}** (Urgency: {item.get('urgency','')})")
                if item.get('risks'):
                    section.append("    - Risks: " + ", ".join(item['risks']))
                if item.get('actions'):
                    section.append("    - Actions: " + ", ".join(item['actions']))
        # Action Items
        action_items = summary.get('action_items', [])
        if action_items:
            section.append("\n### Consolidated Action Items")
            for act in action_items:
                section.append(f"- {act}")
        return '\n'.join(section)
