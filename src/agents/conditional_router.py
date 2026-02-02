"""
Conditional Router Agent

This module contains the ConditionalRouterAgent class responsible for coordinating
between the topic classifier and the appropriate writer agent based on the
classification result. This agent implements the conditional routing logic
that determines which writer agent should handle the content generation.
"""

from typing import Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field

# Import the specialized agents
from agents.topic_classifier import TopicClassifierAgent, TopicClassificationResult
from agents.tech_writer import TechWriterAgent, TechPostResult
from agents.general_writer import GeneralWriterAgent, GeneralPostResult


class WriterType(Enum):
    """
    Enumeration for different types of writer agents.
    
    This enum defines the available writer types that the router
    can route requests to based on topic classification.
    """
    TECH = "Tech"
    GENERAL = "General"


class RoutingDecision(BaseModel):
    """
    Data model for routing decisions.
    
    Attributes:
        topic (str): The original topic provided by the user
        classification_result (TopicClassificationResult): Result from topic classification
        selected_writer (WriterType): The writer agent selected for this topic
        routing_reasoning (str): Explanation for the routing decision
        confidence_score (float): Overall confidence in the routing decision
    """
    topic: str = Field(description="The original topic that was routed")
    classification_result: TopicClassificationResult = Field(description="Result from topic classification")
    selected_writer: WriterType = Field(description="The writer agent selected for this topic")
    routing_reasoning: str = Field(description="Explanation for the routing decision")
    confidence_score: float = Field(description="Overall confidence in the routing decision")


class RoutingResult(BaseModel):
    """
    Data model for the complete routing and generation result.
    
    Attributes:
        routing_decision (RoutingDecision): The routing decision made
        post_result (Union[TechPostResult, GeneralPostResult]): The generated post result
        processing_time_ms (float): Time taken to process the request in milliseconds
        success (bool): Whether the routing and generation was successful
        error_message (Optional[str]): Error message if processing failed
    """
    routing_decision: RoutingDecision = Field(description="The routing decision made")
    post_result: Union[TechPostResult, GeneralPostResult] = Field(description="The generated post result")
    processing_time_ms: float = Field(description="Time taken to process the request in milliseconds")
    success: bool = Field(description="Whether the routing and generation was successful")
    error_message: Optional[str] = Field(description="Error message if processing failed", default=None)


class ConditionalRouterAgent:
    """
    An AI agent that implements conditional routing logic for content generation.
    
    This agent coordinates between the topic classifier and the appropriate writer
    agent based on the classification result. It serves as the central coordinator
    in the AI-Powered LinkedIn Post Generator system.
    
    The routing logic works as follows:
    1. Classify the topic using TopicClassifierAgent
    2. Based on classification, select the appropriate writer agent
    3. Route the request to the selected writer agent
    4. Return the combined routing and generation results
    
    Attributes:
        topic_classifier (TopicClassifierAgent): Agent for classifying topics
        tech_writer (TechWriterAgent): Agent for writing tech posts
        general_writer (GeneralWriterAgent): Agent for writing general posts
        routing_stats (Dict[str, Any]): Statistics about routing decisions
        confidence_threshold (float): Minimum confidence threshold for routing decisions
    """
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 classification_temperature: float = 0.1,
                 writing_temperature: float = 0.7,
                 confidence_threshold: float = 0.6):
        """
        Initialize the Conditional Router Agent.
        
        Args:
            model_name (str): Name of the OpenAI model to use for all agents
            classification_temperature (float): Temperature for topic classification
            writing_temperature (float): Temperature for content writing
            confidence_threshold (float): Minimum confidence for routing decisions
            
        Raises:
            ValueError: If any parameters are invalid
        """
        # Validate inputs
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(classification_temperature, (int, float)) or classification_temperature < 0 or classification_temperature > 2:
            raise ValueError("classification_temperature must be between 0 and 2")
        if not isinstance(writing_temperature, (int, float)) or writing_temperature < 0 or writing_temperature > 2:
            raise ValueError("writing_temperature must be between 0 and 2")
        if not isinstance(confidence_threshold, (int, float)) or confidence_threshold < 0 or confidence_threshold > 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        # Initialize the specialized agents
        self.topic_classifier = TopicClassifierAgent(
            model_name=model_name,
            temperature=classification_temperature
        )
        
        self.tech_writer = TechWriterAgent(
            model_name=model_name,
            temperature=writing_temperature
        )
        
        self.general_writer = GeneralWriterAgent(
            model_name=model_name,
            temperature=writing_temperature
        )
        
        # Set confidence threshold for routing decisions
        self.confidence_threshold = confidence_threshold
        
        # Initialize routing statistics
        self.routing_stats = {
            'total_requests': 0,
            'tech_routes': 0,
            'general_routes': 0,
            'low_confidence_routes': 0,
            'failed_routes': 0,
            'average_processing_time': 0.0
        }
    
    def _make_routing_decision(self, classification_result: TopicClassificationResult) -> RoutingDecision:
        """
        Make a routing decision based on the classification result.
        
        This method implements the core conditional routing logic by analyzing
        the classification result and determining which writer agent should handle
        the content generation.
        
        Args:
            classification_result (TopicClassificationResult): Result from topic classification
            
        Returns:
            RoutingDecision: The routing decision with reasoning
        """
        # Determine the selected writer based on classification
        if classification_result.category == "Tech":
            selected_writer = WriterType.TECH
            routing_reasoning = f"Topic classified as Tech with {classification_result.confidence:.2f} confidence. {classification_result.reasoning}"
        else:
            selected_writer = WriterType.GENERAL
            routing_reasoning = f"Topic classified as General with {classification_result.confidence:.2f} confidence. {classification_result.reasoning}"
        
        # Calculate overall routing confidence
        # If classification confidence is low, we might want to be more conservative
        routing_confidence = classification_result.confidence
        
        # Add logic for edge cases
        if classification_result.confidence < self.confidence_threshold:
            routing_reasoning += " Note: Low confidence classification - using default routing."
            # For low confidence, we might default to General writer as it's more versatile
            if classification_result.category == "Tech" and classification_result.confidence < 0.4:
                selected_writer = WriterType.GENERAL
                routing_reasoning += " Low confidence tech classification - routing to General writer for safety."
                routing_confidence *= 0.8  # Reduce confidence for this override
        
        return RoutingDecision(
            topic=classification_result.topic,
            classification_result=classification_result,
            selected_writer=selected_writer,
            routing_reasoning=routing_reasoning,
            confidence_score=routing_confidence
        )
    
    def _execute_routing(self, routing_decision: RoutingDecision, language: str) -> Union[TechPostResult, GeneralPostResult]:
        """
        Execute the routing by calling the appropriate writer agent.
        
        This method takes the routing decision and calls the corresponding
        writer agent to generate the LinkedIn post.
        
        Args:
            routing_decision (RoutingDecision): The routing decision to execute
            language (str): The language for the post generation
            
        Returns:
            Union[TechPostResult, GeneralPostResult]: The generated post result
            
        Raises:
            RuntimeError: If the routing execution fails
        """
        try:
            if routing_decision.selected_writer == WriterType.TECH:
                # Route to Tech Writer Agent
                return self.tech_writer.generate_tech_post(
                    topic=routing_decision.topic,
                    language=language
                )
            else:
                # Route to General Writer Agent
                return self.general_writer.generate_general_post(
                    topic=routing_decision.topic,
                    language=language
                )
        except Exception as e:
            raise RuntimeError(f"Failed to execute routing to {routing_decision.selected_writer.value} writer: {str(e)}")
    
    def _update_routing_stats(self, routing_decision: RoutingDecision, processing_time: float, success: bool):
        """
        Update routing statistics for monitoring and analytics.
        
        Args:
            routing_decision (RoutingDecision): The routing decision that was made
            processing_time (float): Time taken for processing in milliseconds
            success (bool): Whether the routing was successful
        """
        self.routing_stats['total_requests'] += 1
        
        if routing_decision.selected_writer == WriterType.TECH:
            self.routing_stats['tech_routes'] += 1
        else:
            self.routing_stats['general_routes'] += 1
        
        if routing_decision.confidence_score < self.confidence_threshold:
            self.routing_stats['low_confidence_routes'] += 1
        
        if not success:
            self.routing_stats['failed_routes'] += 1
        
        # Update average processing time
        total_requests = self.routing_stats['total_requests']
        current_avg = self.routing_stats['average_processing_time']
        self.routing_stats['average_processing_time'] = (current_avg * (total_requests - 1) + processing_time) / total_requests
    
    def generate_linkedin_post(self, topic: str, language: str = "English") -> RoutingResult:
        """
        Generate a LinkedIn post using conditional routing.
        
        This is the main method that orchestrates the entire process:
        1. Classify the topic
        2. Make routing decision
        3. Route to appropriate writer
        4. Generate the post
        5. Return comprehensive results
        
        Args:
            topic (str): The topic for the LinkedIn post
            language (str): The language for the post. Defaults to "English"
            
        Returns:
            RoutingResult: Comprehensive result including routing decision and generated post
            
        Raises:
            ValueError: If topic is invalid or language is not supported
            RuntimeError: If the routing or generation process fails
        """
        import time
        start_time = time.time()
        
        try:
            # Validate inputs
            if not topic or not isinstance(topic, str):
                raise ValueError("Topic must be a non-empty string")
            
            if len(topic.strip()) == 0:
                raise ValueError("Topic cannot be empty or whitespace only")
            
            # Step 1: Classify the topic
            classification_result = self.topic_classifier.classify_topic(topic)
            
            # Step 2: Make routing decision
            routing_decision = self._make_routing_decision(classification_result)
            
            # Step 3: Execute routing and generate post
            post_result = self._execute_routing(routing_decision, language)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Update statistics
            self._update_routing_stats(routing_decision, processing_time, True)
            
            return RoutingResult(
                routing_decision=routing_decision,
                post_result=post_result,
                processing_time_ms=processing_time,
                success=True,
                error_message=None
            )
            
        except Exception as e:
            # Calculate processing time even for failures
            processing_time = (time.time() - start_time) * 1000
            
            # Update failure statistics
            if 'routing_decision' in locals():
                self._update_routing_stats(routing_decision, processing_time, False)
            else:
                self.routing_stats['total_requests'] += 1
                self.routing_stats['failed_routes'] += 1
            
            # Create a dummy post result for error cases
            if 'routing_decision' in locals() and routing_decision.selected_writer == WriterType.TECH:
                dummy_post_result = TechPostResult(
                    topic=topic,
                    language=language,
                    post_content=f"Error generating post: {str(e)}",
                    word_count=0,
                    paragraph_count=0,
                    has_call_to_action=False,
                    technical_depth="None"
                )
            else:
                dummy_post_result = GeneralPostResult(
                    topic=topic,
                    language=language,
                    post_content=f"Error generating post: {str(e)}",
                    word_count=0,
                    paragraph_count=0,
                    has_call_to_action=False,
                    content_category="Error"
                )
            
            return RoutingResult(
                routing_decision=routing_decision if 'routing_decision' in locals() else RoutingDecision(
                    selected_writer=WriterType.TECH,
                    confidence_score=0.0,
                    reasoning=f"Error occurred: {str(e)}"
                ),
                post_result=dummy_post_result,
                processing_time_ms=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def batch_generate_posts(self, topics_languages: list[tuple[str, str]]) -> list[RoutingResult]:
        """
        Generate multiple LinkedIn posts using conditional routing.
        
        This method processes multiple topic-language pairs efficiently,
        which is useful for bulk content generation or testing.
        
        Args:
            topics_languages (list[tuple[str, str]]): List of (topic, language) tuples
            
        Returns:
            list[RoutingResult]: List of routing results for each request
            
        Raises:
            ValueError: If topics_languages list is empty or contains invalid items
        """
        if not topics_languages or not isinstance(topics_languages, list):
            raise ValueError("topics_languages must be a non-empty list of (topic, language) tuples")
        
        results = []
        
        for topic, language in topics_languages:
            try:
                result = self.generate_linkedin_post(topic, language)
                results.append(result)
            except Exception as e:
                # Create an error result for failed generations
                error_result = RoutingResult(
                    routing_decision=None,
                    post_result=None,
                    processing_time_ms=0.0,
                    success=False,
                    error_message=f"Batch processing failed for topic '{topic}': {str(e)}"
                )
                results.append(error_result)
        
        return results
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive routing statistics.
        
        Returns:
            Dict[str, Any]: Detailed statistics about routing decisions and performance
        """
        stats = self.routing_stats.copy()
        
        # Calculate additional derived statistics
        if stats['total_requests'] > 0:
            stats['tech_route_percentage'] = (stats['tech_routes'] / stats['total_requests']) * 100
            stats['general_route_percentage'] = (stats['general_routes'] / stats['total_requests']) * 100
            stats['low_confidence_percentage'] = (stats['low_confidence_routes'] / stats['total_requests']) * 100
            stats['failure_rate'] = (stats['failed_routes'] / stats['total_requests']) * 100
        else:
            stats['tech_route_percentage'] = 0.0
            stats['general_route_percentage'] = 0.0
            stats['low_confidence_percentage'] = 0.0
            stats['failure_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """
        Reset all routing statistics to initial values.
        
        This method is useful for testing or when you want to start fresh
        with statistics collection.
        """
        self.routing_stats = {
            'total_requests': 0,
            'tech_routes': 0,
            'general_routes': 0,
            'low_confidence_routes': 0,
            'failed_routes': 0,
            'average_processing_time': 0.0
        }
    
    def analyze_routing_patterns(self) -> Dict[str, Any]:
        """
        Analyze routing patterns and provide insights.
        
        Returns:
            Dict[str, Any]: Analysis of routing patterns and recommendations
        """
        stats = self.get_routing_statistics()
        
        analysis = {
            'patterns': [],
            'recommendations': [],
            'performance_insights': []
        }
        
        # Analyze routing patterns
        if stats['total_requests'] >= 10:  # Only analyze with sufficient data
            if stats['tech_route_percentage'] > 70:
                analysis['patterns'].append("High volume of tech topics detected")
                analysis['recommendations'].append("Consider optimizing tech writer agent for better performance")
            elif stats['general_route_percentage'] > 70:
                analysis['patterns'].append("High volume of general topics detected")
                analysis['recommendations'].append("Consider optimizing general writer agent for better performance")
            
            if stats['low_confidence_percentage'] > 20:
                analysis['patterns'].append("High number of low-confidence classifications")
                analysis['recommendations'].append("Consider improving topic classifier or adjusting confidence threshold")
            
            if stats['failure_rate'] > 10:
                analysis['patterns'].append("High failure rate detected")
                analysis['recommendations'].append("Review error handling and input validation")
        
        # Performance insights
        if stats['average_processing_time'] > 5000:  # More than 5 seconds
            analysis['performance_insights'].append("Processing time is above optimal range")
        elif stats['average_processing_time'] < 1000:  # Less than 1 second
            analysis['performance_insights'].append("Excellent processing performance")
        
        return analysis
