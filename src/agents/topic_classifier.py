"""
Topic Classifier Agent

This module contains the TopicClassifierAgent class responsible for analyzing
user-provided topics and classifying them as either "Tech" or "General" topics.
This classification determines which writer agent will handle the content generation.
"""

from typing import Dict, Any
from langchain.schema import BasePromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
import os


class TopicClassificationResult(BaseModel):
    """
    Data model for topic classification results.
    
    Attributes:
        topic (str): The original topic provided by the user
        category (str): Either "Tech" or "General" classification
        confidence (float): Confidence score of the classification (0.0 to 1.0)
        reasoning (str): Explanation for why the topic was classified this way
    """
    topic: str = Field(description="The original topic that was classified")
    category: str = Field(description="Classification result: either 'Tech' or 'General'")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Explanation for the classification decision")


class TopicClassifierAgent:
    """
    An AI agent that classifies topics into Tech or General categories.
    
    This agent uses a Large Language Model (LLM) to analyze the given topic
    and determine whether it's technology-related or general in nature.
    The classification is used to route the topic to the appropriate writer agent.
    
    Attributes:
        llm (ChatOpenAI): The language model for classification
        classification_chain (LLMChain): The LangChain for processing topics
        tech_keywords (list): List of technology-related keywords for quick filtering
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.1):
        """
        Initialize the Topic Classifier Agent.
        
        Args:
            model_name (str): Name of the OpenAI model to use. Defaults to "gpt-3.5-turbo"
            temperature (float): Temperature for LLM generation. Lower values for more deterministic output
        
        Raises:
            ValueError: If model_name is invalid or temperature is out of range
        """
        # Validate inputs
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            raise ValueError("temperature must be a number between 0 and 2")
        
        # Initialize the language model with parameters optimized for classification
        self.llm = Ollama(
            model=model_name if model_name != "gpt-3.5-turbo" else "llama3.2:3b",
            temperature=temperature  # Lower temperature for more consistent classification
        )
        
        # Define the classification prompt template
        # This prompt guides the LLM to classify topics consistently
        self.classification_prompt = PromptTemplate(
            input_variables=["topic"],
            template="""You are an expert topic classifier for LinkedIn content generation.

Your task is to analyze the given topic and classify it as either "Tech" or "General".

TECH topics include:
- Artificial Intelligence, Machine Learning, Deep Learning
- Software Development, Programming, Coding
- Cloud Computing, DevOps, Infrastructure
- Cybersecurity, Data Privacy
- Blockchain, Cryptocurrency
- Mobile Apps, Web Development
- IoT (Internet of Things)
- Robotics, Automation
- Data Science, Analytics
- Technology trends, innovation
- Digital transformation

GENERAL topics include:
- Business Strategy, Leadership
- Marketing, Sales
- Human Resources, Workplace culture
- Finance, Economics
- Education, Learning
- Health, Wellness
- Environment, Sustainability
- Social issues, Politics
- Arts, Culture, Entertainment
- Travel, Lifestyle
- Personal development
- Industry news (non-technical)

Topic to classify: "{topic}"

Provide your classification in this exact format:
Category: [Tech/General]
Confidence: [0.0-1.0]
Reasoning: [Brief explanation of your decision]

Be decisive and accurate in your classification."""
        )
        
        # Create the classification chain
        self.classification_chain = LLMChain(
            llm=self.llm,
            prompt=self.classification_prompt
        )
        
        # Technology keywords for quick pre-filtering optimization
        self.tech_keywords = [
            'ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning',
            'software', 'programming', 'coding', 'developer', 'app', 'web development',
            'cloud', 'aws', 'azure', 'gcp', 'devops', 'infrastructure',
            'cybersecurity', 'security', 'data privacy', 'blockchain', 'cryptocurrency',
            'mobile', 'iot', 'robotics', 'automation', 'data science', 'analytics',
            'technology', 'tech', 'digital', 'innovation', 'algorithm', 'api'
        ]
    
    def _quick_keyword_check(self, topic: str) -> str:
        """
        Perform a quick keyword-based classification for optimization.
        
        This method checks if the topic contains obvious tech keywords
        for faster classification when the answer is clear.
        
        Args:
            topic (str): The topic to analyze
            
        Returns:
            str: "Tech" if tech keywords found, "Uncertain" otherwise
        """
        topic_lower = topic.lower()
        
        # Check for any tech keywords in the topic
        for keyword in self.tech_keywords:
            if keyword in topic_lower:
                return "Tech"
        
        return "Uncertain"
    
    def _parse_classification_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response into structured data.
        
        Args:
            response (str): The raw response from the LLM
            
        Returns:
            Dict[str, Any]: Parsed classification data
            
        Raises:
            ValueError: If response format is invalid or missing required fields
        """
        lines = response.strip().split('\n')
        result = {
            'category': None,
            'confidence': None,
            'reasoning': None
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith('Category:'):
                result['category'] = line.split(':', 1)[1].strip()
            elif line.startswith('Confidence:'):
                try:
                    result['confidence'] = float(line.split(':', 1)[1].strip())
                except ValueError:
                    result['confidence'] = 0.5  # Default confidence
            elif line.startswith('Reasoning:'):
                result['reasoning'] = line.split(':', 1)[1].strip()
        
        # Validate required fields
        if not result['category'] or result['category'] not in ['Tech', 'General']:
            raise ValueError("Invalid or missing category in classification response")
        
        if result['confidence'] is None:
            result['confidence'] = 0.5  # Default confidence
        
        if not result['reasoning']:
            result['reasoning'] = "No reasoning provided"
        
        return result
    
    def classify_topic(self, topic: str) -> TopicClassificationResult:
        """
        Classify a given topic as Tech or General.
        
        This is the main method that performs the topic classification.
        It first attempts a quick keyword check, then uses the LLM for detailed analysis.
        
        Args:
            topic (str): The topic to classify
            
        Returns:
            TopicClassificationResult: Object containing classification results
            
        Raises:
            ValueError: If topic is empty or invalid
            RuntimeError: If LLM classification fails
        """
        # Validate input
        if not topic or not isinstance(topic, str):
            raise ValueError("Topic must be a non-empty string")
        
        if len(topic.strip()) == 0:
            raise ValueError("Topic cannot be empty or whitespace only")
        
        # Perform quick keyword check for optimization
        quick_result = self._quick_keyword_check(topic)
        
        # If quick check is certain, we can skip LLM call
        if quick_result == "Tech":
            return TopicClassificationResult(
                topic=topic,
                category="Tech",
                confidence=0.9,  # High confidence for keyword matches
                reasoning="Topic contains clear technology-related keywords"
            )
        
        # Generate classification using the LLM chain
        try:
            response = self.classification_chain.invoke({"topic": topic})
            # Extract the text from the response dictionary
            if isinstance(response, dict):
                response_text = response.get('text', str(response))
            else:
                response_text = str(response)
            parsed_result = self._parse_classification_response(response_text)
                
            return TopicClassificationResult(
                topic=topic,
                category=parsed_result['category'],
                confidence=parsed_result['confidence'],
                reasoning=parsed_result['reasoning']
            )
            
        except Exception as e:
            # Fallback classification in case of LLM failure
            raise RuntimeError(f"Failed to classify topic: {str(e)}")
    
    def batch_classify_topics(self, topics: list) -> list[TopicClassificationResult]:
        """
        Classify multiple topics in batch.
        
        This method processes multiple topics efficiently, which is useful
        for testing or bulk processing scenarios.
        
        Args:
            topics (list): List of topics to classify
            
        Returns:
            list[TopicClassificationResult]: List of classification results
            
        Raises:
            ValueError: If topics list is empty or contains invalid items
        """
        if not topics or not isinstance(topics, list):
            raise ValueError("Topics must be a non-empty list")
        
        results = []
        
        for topic in topics:
            try:
                result = self.classify_topic(topic)
                results.append(result)
            except Exception as e:
                # Create an error result for failed classifications
                error_result = TopicClassificationResult(
                    topic=topic,
                    category="General",  # Default fallback
                    confidence=0.0,
                    reasoning=f"Classification failed: {str(e)}"
                )
                results.append(error_result)
        
        return results
    
    def get_classification_stats(self, results: list[TopicClassificationResult]) -> Dict[str, Any]:
        """
        Generate statistics for a batch of classification results.
        
        Args:
            results (list[TopicClassificationResult]): List of classification results
            
        Returns:
            Dict[str, Any]: Statistics including counts, percentages, and average confidence
        """
        if not results:
            return {
                'total_topics': 0,
                'tech_count': 0,
                'general_count': 0,
                'tech_percentage': 0.0,
                'general_percentage': 0.0,
                'average_confidence': 0.0
            }
        
        tech_count = sum(1 for r in results if r.category == 'Tech')
        general_count = len(results) - tech_count
        total_confidence = sum(r.confidence for r in results)
        
        return {
            'total_topics': len(results),
            'tech_count': tech_count,
            'general_count': general_count,
            'tech_percentage': (tech_count / len(results)) * 100,
            'general_percentage': (general_count / len(results)) * 100,
            'average_confidence': total_confidence / len(results)
        }
