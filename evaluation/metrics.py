"""
Evaluation metrics module for RAG system performance assessment.
Provides comprehensive metrics for retrieval quality, answer quality, and system performance.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import time
from datetime import datetime
import json
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Comprehensive evaluator for RAG systems."""
    
    def __init__(self):
        """Initialize RAG evaluator."""
        self.evaluation_history = []
    
    def calculate_retrieval_metrics(self, 
                                retrieved_docs: List[Dict[str, Any]], 
                                query: str) -> Dict[str, float]:
        """
        Calculate retrieval-specific metrics.
        
        Args:
            retrieved_docs: List of retrieved documents
            query: Original query
            
        Returns:
            Dictionary of retrieval metrics
        """
        if not retrieved_docs:
            return {
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "f1_at_k": 0.0,
                "mean_similarity": 0.0,
                "similarity_std": 0.0
            }
        
        # Similarity-based metrics
        similarities = [doc.get("similarity", 0.0) for doc in retrieved_docs]
        
        metrics = {
            "precision_at_k": self._calculate_precision_at_k(retrieved_docs, query),
            "recall_at_k": self._calculate_recall_at_k(retrieved_docs, query),
            "f1_at_k": self._calculate_f1_at_k(retrieved_docs, query),
            "mean_similarity": np.mean(similarities),
            "similarity_std": np.std(similarities),
            "max_similarity": np.max(similarities),
            "min_similarity": np.min(similarities)
        }
        
        return metrics
    
    def _calculate_precision_at_k(self, docs: List[Dict[str, Any]], query: str) -> float:
        """Calculate precision@k based on similarity threshold."""
        if not docs:
            return 0.0
        
        # Consider relevant if similarity > 0.1 (adjustable threshold)
        relevant_count = sum(1 for doc in docs if doc.get("similarity", 0) > 0.1)
        return relevant_count / len(docs)
    
    def _calculate_recall_at_k(self, docs: List[Dict[str, Any]], query: str) -> float:
        """Calculate recall@k (simplified for RAG context)."""
        if not docs:
            return 0.0
        
        # For RAG, we use similarity-based proxy for recall
        high_similarity_count = sum(1 for doc in docs if doc.get("similarity", 0) > 0.2)
        return min(high_similarity_count / 5, 1.0)  # Assume 5 relevant docs max
    
    def _calculate_f1_at_k(self, docs: List[Dict[str, Any]], query: str) -> float:
        """Calculate F1@k score."""
        precision = self._calculate_precision_at_k(docs, query)
        recall = self._calculate_recall_at_k(docs, query)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_answer_quality_metrics(self, 
                                   answer: str, 
                                   query: str, 
                                   sources: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate answer quality metrics.
        
        Args:
            answer: Generated answer
            query: Original query
            sources: Source documents
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {
            "length_score": self._calculate_length_score(answer),
            "relevance_score": self._calculate_relevance_score(answer, query),
            "citation_score": self._calculate_citation_score(sources),
            "coherence_score": self._calculate_coherence_score(answer),
            "completeness_score": self._calculate_completeness_score(answer, query)
        }
        
        # Calculate overall score
        weights = {
            "length_score": 0.15,
            "relevance_score": 0.30,
            "citation_score": 0.25,
            "coherence_score": 0.20,
            "completeness_score": 0.10
        }
        
        metrics["overall_score"] = sum(metrics[key] * weights[key] for key in weights)
        
        return metrics
    
    def _calculate_length_score(self, answer: str) -> float:
        """Calculate length appropriateness score."""
        word_count = len(answer.split())
        
        if word_count < 10:
            return 0.2  # Too short
        elif word_count > 500:
            return 0.7  # Too long
        elif 50 <= word_count <= 200:
            return 1.0  # Good length
        else:
            return 0.8  # Acceptable
    
    def _calculate_relevance_score(self, answer: str, query: str) -> float:
        """Calculate relevance score using keyword overlap."""
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(answer_words))
        return min(overlap / len(query_words), 1.0)
    
    def _calculate_citation_score(self, sources: List[Dict[str, Any]]) -> float:
        """Calculate citation quality score."""
        if not sources:
            return 0.0
        
        # Check for proper citation elements
        citation_elements = 0
        max_elements = len(sources) * 3  # filename, page, similarity per source
        
        for source in sources:
            if source.get("filename"):
                citation_elements += 1
            if source.get("page"):
                citation_elements += 1
            if source.get("similarity") is not None:
                citation_elements += 1
        
        return min(citation_elements / max_elements, 1.0)
    
    def _calculate_coherence_score(self, answer: str) -> float:
        """Calculate basic coherence score."""
        sentences = answer.split('.')
        if len(sentences) < 2:
            return 0.5
        
        # Simple coherence: check for sentence variety
        unique_starts = len(set(sentence.strip().split()[:2] for sentence in sentences if sentence.strip()))
        return min(unique_starts / len(sentences), 1.0)
    
    def _calculate_completeness_score(self, answer: str, query: str) -> float:
        """Calculate answer completeness score."""
        # Check for question words in answer
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        if not query_words:
            return 0.0
        
        coverage = len(query_words.intersection(answer_words)) / len(query_words)
        
        # Bonus for longer answers that might be more complete
        length_bonus = min(len(answer.split()) / 100, 1.0) * 0.2
        
        return min(coverage + length_bonus, 1.0)
    
    def evaluate_single_query(self, 
                          query_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single query result comprehensively.
        
        Args:
            query_result: Result from RAG system
            
        Returns:
            Comprehensive evaluation dictionary
        """
        evaluation = {
            "query": query_result.get("query", ""),
            "answer": query_result.get("answer", ""),
            "sources": query_result.get("sources", []),
            "retrieval_time": query_result.get("retrieval_time", 0),
            "error": query_result.get("error"),
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate retrieval metrics
        if evaluation["sources"]:
            retrieval_metrics = self.calculate_retrieval_metrics(
                evaluation["sources"], 
                evaluation["query"]
            )
            evaluation.update(retrieval_metrics)
        
        # Calculate answer quality metrics
        if evaluation["answer"]:
            quality_metrics = self.calculate_answer_quality_metrics(
                evaluation["answer"],
                evaluation["query"],
                evaluation["sources"]
            )
            evaluation.update(quality_metrics)
        
        # Success/failure status
        evaluation["success"] = (
            evaluation["error"] is None and 
            len(evaluation["answer"]) > 0 and
            evaluation.get("overall_score", 0) > 0.3
        )
        
        return evaluation
    
    def evaluate_batch(self, 
                    query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a batch of query results.
        
        Args:
            query_results: List of RAG query results
            
        Returns:
            Batch evaluation summary
        """
        individual_evaluations = []
        
        for result in query_results:
            eval_result = self.evaluate_single_query(result)
            individual_evaluations.append(eval_result)
        
        # Calculate aggregate metrics
        successful_evals = [e for e in individual_evaluations if e.get("success", False)]
        
        summary = {
            "total_queries": len(individual_evaluations),
            "successful_queries": len(successful_evals),
            "failed_queries": len(individual_evaluations) - len(successful_evals),
            "success_rate": len(successful_evals) / len(individual_evaluations) if individual_evaluations else 0,
            "average_retrieval_time": np.mean([e.get("retrieval_time", 0) for e in individual_evaluations]),
            "average_quality_score": np.mean([e.get("overall_score", 0) for e in successful_evals]) if successful_evals else 0,
            "average_precision": np.mean([e.get("precision_at_k", 0) for e in successful_evals]) if successful_evals else 0,
            "average_recall": np.mean([e.get("recall_at_k", 0) for e in successful_evals]) if successful_evals else 0,
            "average_f1": np.mean([e.get("f1_at_k", 0) for e in successful_evals]) if successful_evals else 0,
            "evaluations": individual_evaluations,
            "timestamp": datetime.now().isoformat()
        }
        
        return summary
    
    def generate_evaluation_report(self, 
                            evaluation_summary: Dict[str, Any]) -> str:
        """
        Generate a human-readable evaluation report.
        
        Args:
            evaluation_summary: Summary from evaluate_batch
            
        Returns:
            Formatted report string
        """
        report = f"""
# RAG System Evaluation Report
Generated: {evaluation_summary['timestamp']}

## Executive Summary
- Total Queries: {evaluation_summary['total_queries']}
- Success Rate: {evaluation_summary['success_rate']*100:.1f}%
- Average Retrieval Time: {evaluation_summary['average_retrieval_time']:.2f}s
- Average Quality Score: {evaluation_summary['average_quality_score']:.3f}

## Performance Metrics
- Average Precision@K: {evaluation_summary['average_precision']:.3f}
- Average Recall@K: {evaluation_summary['average_recall']:.3f}
- Average F1@K: {evaluation_summary['average_f1']:.3f}

## Query Analysis
- Successful Queries: {evaluation_summary['successful_queries']}
- Failed Queries: {evaluation_summary['failed_queries']}

## Recommendations
"""
        
        # Add recommendations based on performance
        if evaluation_summary['success_rate'] < 0.8:
            report += "- IMPROVEMENT NEEDED: Success rate below 80%\\n"
        
        if evaluation_summary['average_retrieval_time'] > 30:
            report += "- OPTIMIZATION NEEDED: Average retrieval time exceeds 30 seconds\\n"
        
        if evaluation_summary['average_quality_score'] < 0.6:
            report += "- QUALITY IMPROVEMENT: Average quality score below 0.6\\n"
        
        if evaluation_summary['success_rate'] >= 0.8 and evaluation_summary['average_retrieval_time'] <= 30:
            report += "- GOOD PERFORMANCE: System meets quality and speed targets\\n"
        
        return report
    
    def save_evaluation_results(self, 
                           evaluation_summary: Dict[str, Any], 
                           filename_prefix: str = "rag_evaluation") -> None:
        """
        Save evaluation results to files.
        
        Args:
            evaluation_summary: Evaluation summary
            filename_prefix: Prefix for output files
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        detailed_file = f"{filename_prefix}_detailed_{timestamp}.json"
        with open(detailed_file, 'w') as f:
            json.dump(evaluation_summary['evaluations'], f, indent=2)
        
        # Save summary report
        report = self.generate_evaluation_report(evaluation_summary)
        report_file = f"{filename_prefix}_summary_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save CSV summary
        if evaluation_summary['evaluations']:
            df = pd.DataFrame(evaluation_summary['evaluations'])
            csv_file = f"{filename_prefix}_summary_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
        
        logger.info(f"Evaluation results saved with prefix: {filename_prefix}_{timestamp}")


if __name__ == "__main__":
    # Example usage
    evaluator = RAGEvaluator()
    
    # Example query results
    example_results = [
        {
            "query": "What are main challenges in global development?",
            "answer": "Based on the World Bank reports, main challenges include macroeconomic implications, environmental impact, and technological change.",
            "sources": [
                {"filename": "WDR 2020.pdf", "page": 89, "similarity": 0.85},
                {"filename": "WDR 2025.pdf", "page": 1, "similarity": 0.78}
            ],
            "retrieval_time": 15.2
        },
        {
            "query": "How does climate change affect developing countries?",
            "answer": "Climate change significantly affects developing countries through increased extreme weather events, agricultural impacts, and water scarcity.",
            "sources": [
                {"filename": "WDR 2021.pdf", "page": 45, "similarity": 0.92}
            ],
            "retrieval_time": 12.8
        }
    ]
    
    # Evaluate batch
    summary = evaluator.evaluate_batch(example_results)
    
    # Generate and print report
    report = evaluator.generate_evaluation_report(summary)
    print(report)
    
    # Save results
    evaluator.save_evaluation_results(summary)
