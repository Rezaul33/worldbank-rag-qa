"""
Answer generator module that provides additional formatting and post-processing
for RAG responses, including citation formatting and answer quality checks.
"""

import re
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Enhances and formats RAG-generated answers."""
    
    def __init__(self):
        """Initialize answer generator."""
        pass
    
    def format_answer_with_citations(self, 
                                   answer: str, 
                                   sources: List[Dict[str, Any]]) -> str:
        """
        Format answer with proper citations.
        
        Args:
            answer: Generated answer
            sources: List of source documents
            
        Returns:
            Formatted answer with citations
        """
        if not sources:
            return answer
        
        # Create source mapping
        source_map = {}
        for i, source in enumerate(sources, 1):
            key = f"[{i}]"
            source_map[key] = f"{source['filename']} (Page {source['page']})"
        
        # Add citations section
        citations = "\n\n**Sources:**\n"
        for key, citation in source_map.items():
            citations += f"{key} {citation}\n"
        
        formatted_answer = answer + citations
        return formatted_answer
    
    def extract_key_points(self, answer: str) -> List[str]:
        """
        Extract key points from the answer.
        
        Args:
            answer: Generated answer
            
        Returns:
            List of key points
        """
        # Split by common list patterns
        patterns = [
            r'\d+\.\s+(.*?)(?=\n\d+\.|\n\n|$)',  # Numbered lists
            r'[-•]\s+(.*?)(?=\n[-•]|\n\n|$)',   # Bullet points
            r'(.*?)(?=\n\n|$)'                   # Paragraphs
        ]
        
        key_points = []
        
        for pattern in patterns:
            matches = re.findall(pattern, answer, re.MULTILINE | re.DOTALL)
            for match in matches:
                point = match.strip()
                if len(point) > 20 and point not in key_points:
                    key_points.append(point)
        
        # If no structured points found, split by paragraphs
        if not key_points:
            paragraphs = [p.strip() for p in answer.split('\n\n') if p.strip()]
            key_points = paragraphs[:5]  # Limit to first 5 paragraphs
        
        return key_points
    
    def assess_answer_quality(self, 
                             answer: str, 
                             query: str, 
                             sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess the quality of the generated answer.
        
        Args:
            answer: Generated answer
            query: Original query
            sources: Source documents
            
        Returns:
            Quality assessment dictionary
        """
        assessment = {
            "length_score": 0,
            "relevance_score": 0,
            "citation_score": 0,
            "overall_score": 0,
            "issues": []
        }
        
        # Length assessment
        answer_length = len(answer.split())
        if answer_length < 20:
            assessment["issues"].append("Answer is too short")
            assessment["length_score"] = 0.2
        elif answer_length > 500:
            assessment["issues"].append("Answer is very long")
            assessment["length_score"] = 0.7
        else:
            assessment["length_score"] = 1.0
        
        # Relevance assessment (simple keyword matching)
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(query_words.intersection(answer_words))
        assessment["relevance_score"] = min(overlap / len(query_words), 1.0)
        
        # Citation assessment
        if sources:
            assessment["citation_score"] = 1.0
        else:
            assessment["issues"].append("No sources provided")
            assessment["citation_score"] = 0.0
        
        # Overall score
        assessment["overall_score"] = (
            assessment["length_score"] * 0.3 +
            assessment["relevance_score"] * 0.4 +
            assessment["citation_score"] * 0.3
        )
        
        return assessment
    
    def improve_answer(self, 
                      original_answer: str, 
                      query: str, 
                      sources: List[Dict[str, Any]]) -> str:
        """
        Improve the answer based on quality assessment.
        
        Args:
            original_answer: Original generated answer
            query: Original query
            sources: Source documents
            
        Returns:
            Improved answer
        """
        assessment = self.assess_answer_quality(original_answer, query, sources)
        
        improved_answer = original_answer
        
        # Add clarification if answer is too short
        if assessment["length_score"] < 0.5:
            improved_answer += "\n\nNote: This answer is based on the available context. For more comprehensive information, please consider asking a more specific question."
        
        # Add source information if missing
        if not sources and assessment["citation_score"] == 0:
            improved_answer += "\n\nSource: World Bank Development Reports"
        
        # Format with citations
        if sources:
            improved_answer = self.format_answer_with_citations(improved_answer, sources)
        
        return improved_answer
    
    def generate_summary(self, answer: str) -> str:
        """
        Generate a brief summary of the answer.
        
        Args:
            answer: Full answer
            
        Returns:
            Summary (max 100 words)
        """
        # Extract first few sentences
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        summary = ""
        word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= 100:
                summary += sentence + ". "
                word_count += sentence_words
            else:
                break
        
        if not summary:
            summary = answer[:200] + "..." if len(answer) > 200 else answer
        
        return summary.strip()


if __name__ == "__main__":
    # Example usage
    generator = AnswerGenerator()
    
    # Test answer
    test_answer = """
    Climate change significantly affects developing countries through various mechanisms. 
    First, it increases the frequency and intensity of extreme weather events such as hurricanes, droughts, and floods. 
    Second, it impacts agricultural productivity, threatening food security. 
    Third, it exacerbates water scarcity issues. 
    Fourth, it can lead to displacement of populations and migration pressures.
    """
    
    test_sources = [
        {"filename": "World Development Report 2020.pdf", "page": 45, "similarity": 0.85},
        {"filename": "World Development Report 2021.pdf", "page": 23, "similarity": 0.78}
    ]
    
    test_query = "How does climate change affect developing countries?"
    
    # Test quality assessment
    quality = generator.assess_answer_quality(test_answer, test_query, test_sources)
    print("Quality Assessment:")
    for key, value in quality.items():
        print(f"{key}: {value}")
    
    # Test formatting
    formatted = generator.format_answer_with_citations(test_answer, test_sources)
    print(f"\nFormatted Answer:\n{formatted}")
    
    # Test key points extraction
    key_points = generator.extract_key_points(test_answer)
    print(f"\nKey Points:")
    for i, point in enumerate(key_points, 1):
        print(f"{i}. {point}")
    
    # Test summary
    summary = generator.generate_summary(test_answer)
    print(f"\nSummary: {summary}")
