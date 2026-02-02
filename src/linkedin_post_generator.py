"""
AI-Powered LinkedIn Post Generator - Main System

This module contains the main LinkedInPostGenerator class that serves as the
primary interface for the AI-Powered LinkedIn Post Generator system.
It integrates all the specialized agents and provides a clean, user-friendly API
for generating LinkedIn posts with conditional routing and multi-language support.
"""

import os
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Import the specialized agents
from agents.conditional_router import ConditionalRouterAgent, RoutingResult
from agents.topic_classifier import TopicClassificationResult
from agents.tech_writer import TechWriterAgent, TechPostResult
from agents.general_writer import GeneralWriterAgent, GeneralPostResult


class GenerationRequest(BaseModel):
    """
    Data model for a LinkedIn post generation request.
    
    Attributes:
        topic (str): The topic for the LinkedIn post
        language (str): The language for the post (default: "English")
        user_preferences (Optional[Dict[str, Any]]): Additional user preferences
        request_id (Optional[str]): Unique identifier for the request
    """
    topic: str = Field(description="The topic for the LinkedIn post", min_length=1, max_length=500)
    language: str = Field(description="The language for the post", default="English")
    user_preferences: Optional[Dict[str, Any]] = Field(default=None, description="Additional user preferences")
    request_id: Optional[str] = Field(default=None, description="Unique identifier for the request")
    
    @validator('language')
    def validate_language(cls, v):
        """Validate that the language is supported."""
        supported_languages = [
            'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese',
            'Dutch', 'Russian', 'Chinese', 'Japanese', 'Korean', 'Arabic',
            'Hindi', 'Bengali', 'Tamil', 'Telugu', 'Marathi', 'Gujarati'
        ]
        if v.title() not in supported_languages:
            raise ValueError(f"Language '{v}' is not supported. Supported languages: {', '.join(supported_languages)}")
        return v.title()


class GenerationResponse(BaseModel):
    """
    Data model for a LinkedIn post generation response.
    
    Attributes:
        request (GenerationRequest): The original request
        routing_result (RoutingResult): The routing and generation result
        timestamp (datetime): When the response was generated
        success (bool): Whether the generation was successful
        error_message (Optional[str]): Error message if generation failed
        metadata (Dict[str, Any]): Additional metadata about the generation
    """
    request: GenerationRequest = Field(description="The original generation request")
    routing_result: Optional[RoutingResult] = Field(description="The routing and generation result")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the response was generated")
    success: bool = Field(description="Whether the generation was successful")
    error_message: Optional[str] = Field(default=None, description="Error message if generation failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the generation")


class LinkedInPostGenerator:
    """
    Main AI-Powered LinkedIn Post Generator system.
    
    This class serves as the primary interface for the entire AI agent system.
    It integrates the topic classifier, conditional router, and writer agents
    to provide a seamless experience for generating LinkedIn posts with
    conditional routing and multi-language support.
    
    Key Features:
    - Automatic topic classification (Tech vs General)
    - Conditional routing to appropriate writer agent
    - Multi-language support for content generation
    - Comprehensive error handling and logging
    - Performance monitoring and statistics
    - Batch processing capabilities
    - User preference customization
    
    Attributes:
        router (ConditionalRouterAgent): The conditional router agent
        default_language (str): Default language for content generation
        statistics (Dict[str, Any]): System-wide statistics
        config (Dict[str, Any]): System configuration
    """
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 classification_temperature: float = 0.1,
                 writing_temperature: float = 0.7,
                 confidence_threshold: float = 0.6,
                 default_language: str = "English",
                 enable_statistics: bool = True,
                 config_file: Optional[str] = None):
        """
        Initialize the LinkedIn Post Generator system.
        
        Args:
            model_name (str): Name of the OpenAI model to use for all agents
            classification_temperature (float): Temperature for topic classification
            writing_temperature (float): Temperature for content writing
            confidence_threshold (float): Minimum confidence for routing decisions
            default_language (str): Default language for content generation
            enable_statistics (bool): Whether to enable statistics collection
            config_file (Optional[str]): Path to configuration file
            
        Raises:
            ValueError: If any parameters are invalid
            EnvironmentError: If required environment variables are not set
        """
        # Load environment variables
        load_dotenv()
        
        # Validate OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY environment variable is required. Please set it in your environment or .env file.")
        
        # Validate parameters
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(default_language, str) or not default_language:
            raise ValueError("default_language must be a non-empty string")
        
        # Initialize the conditional router
        self.router = ConditionalRouterAgent(
            model_name=model_name,
            classification_temperature=classification_temperature,
            writing_temperature=writing_temperature,
            confidence_threshold=confidence_threshold
        )
        
        # Set default language
        self.default_language = default_language.title()
        
        # Initialize statistics
        self.enable_statistics = enable_statistics
        self.statistics = {
            'total_requests': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'languages_used': {},
            'topics_processed': [],
            'average_generation_time': 0.0,
            'system_start_time': datetime.now().isoformat()
        } if enable_statistics else None
        
        # Load configuration if provided
        self.config = self._load_config(config_file) if config_file else {}
        
        # System metadata
        self.system_info = {
            'version': '1.0.0',
            'model_name': model_name,
            'supported_languages': [
                'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese',
                'Dutch', 'Russian', 'Chinese', 'Japanese', 'Korean', 'Arabic',
                'Hindi', 'Bengali', 'Tamil', 'Telugu', 'Marathi', 'Gujarati'
            ],
            'features': [
                'Automatic topic classification',
                'Conditional routing',
                'Multi-language support',
                'Performance monitoring',
                'Batch processing'
            ]
        }
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_file (str): Path to the configuration file
            
        Returns:
            Dict[str, Any]: Loaded configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is not valid JSON
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in configuration file: {e}")
    
    def _update_statistics(self, request: GenerationRequest, response: GenerationResponse):
        """
        Update system statistics for monitoring and analytics.
        
        Args:
            request (GenerationRequest): The generation request
            response (GenerationResponse): The generation response
        """
        if not self.enable_statistics or not self.statistics:
            return
        
        self.statistics['total_requests'] += 1
        
        if response.success:
            self.statistics['successful_generations'] += 1
        else:
            self.statistics['failed_generations'] += 1
        
        # Track language usage
        language = request.language
        if language not in self.statistics['languages_used']:
            self.statistics['languages_used'][language] = 0
        self.statistics['languages_used'][language] += 1
        
        # Track topics processed
        self.statistics['topics_processed'].append({
            'topic': request.topic,
            'language': language,
            'timestamp': response.timestamp.isoformat(),
            'success': response.success
        })
        
        # Update average generation time
        if response.routing_result and response.routing_result.processing_time_ms:
            total_requests = self.statistics['total_requests']
            current_avg = self.statistics['average_generation_time']
            new_time = response.routing_result.processing_time_ms
            self.statistics['average_generation_time'] = (current_avg * (total_requests - 1) + new_time) / total_requests
    
    def generate_post(self, 
                     topic: str, 
                     language: Optional[str] = None,
                     user_preferences: Optional[Dict[str, Any]] = None) -> GenerationResponse:
        """
        Generate a LinkedIn post using the AI agent system.
        
        This is the main method that orchestrates the entire post generation process.
        It handles topic classification, conditional routing, and content generation
        in a seamless, user-friendly interface.
        
        Args:
            topic (str): The topic for the LinkedIn post
            language (Optional[str]): The language for the post (uses default if not specified)
            user_preferences (Optional[Dict[str, Any]]): Additional user preferences
            
        Returns:
            GenerationResponse: Comprehensive response with generated post and metadata
            
        Raises:
            ValueError: If topic is invalid or language is not supported
            RuntimeError: If the generation process fails
        """
        # Use default language if not specified
        if language is None:
            language = self.default_language
        
        # Create request object
        request = GenerationRequest(
            topic=topic,
            language=language,
            user_preferences=user_preferences,
            request_id=f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        )
        
        try:
            # Generate post using the router
            routing_result = self.router.generate_linkedin_post(topic, language)
            
            # Create response
            response = GenerationResponse(
                request=request,
                routing_result=routing_result,
                success=routing_result.success,
                error_message=routing_result.error_message if not routing_result.success else None,
                metadata={
                    'system_info': self.system_info,
                    'router_stats': self.router.get_routing_statistics(),
                    'user_preferences': user_preferences
                }
            )
            
            # Update statistics
            self._update_statistics(request, response)
            
            return response
            
        except Exception as e:
            # Create error response
            error_response = GenerationResponse(
                request=request,
                routing_result=None,
                success=False,
                error_message=str(e),
                metadata={
                    'system_info': self.system_info,
                    'error_type': type(e).__name__
                }
            )
            
            # Update statistics
            self._update_statistics(request, error_response)
            
            return error_response
    
    def batch_generate_posts(self, requests: List[Union[GenerationRequest, Dict[str, Any]]]) -> List[GenerationResponse]:
        """
        Generate multiple LinkedIn posts in batch.
        
        This method processes multiple generation requests efficiently,
        which is useful for content creation campaigns or testing.
        
        Args:
            requests (List[Union[GenerationRequest, Dict[str, Any]]]): List of generation requests
            
        Returns:
            List[GenerationResponse]: List of generation responses
            
        Raises:
            ValueError: If requests list is empty or contains invalid items
        """
        if not requests or not isinstance(requests, list):
            raise ValueError("requests must be a non-empty list")
        
        responses = []
        
        for req in requests:
            try:
                # Convert dict to GenerationRequest if needed
                if isinstance(req, dict):
                    generation_req = GenerationRequest(**req)
                elif isinstance(req, GenerationRequest):
                    generation_req = req
                else:
                    raise ValueError(f"Invalid request type: {type(req)}")
                
                # Generate post
                response = self.generate_post(
                    topic=generation_req.topic,
                    language=generation_req.language,
                    user_preferences=generation_req.user_preferences
                )
                
                responses.append(response)
                
            except Exception as e:
                # Create error response
                error_response = GenerationResponse(
                    request=generation_req if 'generation_req' in locals() else GenerationRequest(topic="Unknown", language="English"),
                    routing_result=None,
                    success=False,
                    error_message=f"Batch processing failed: {str(e)}"
                )
                responses.append(error_response)
        
        return responses
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dict[str, Any]: Detailed statistics about system performance and usage
        """
        if not self.enable_statistics or not self.statistics:
            return {'message': 'Statistics collection is disabled'}
        
        stats = self.statistics.copy()
        
        # Calculate additional derived statistics
        if stats['total_requests'] > 0:
            stats['success_rate'] = (stats['successful_generations'] / stats['total_requests']) * 100
            stats['failure_rate'] = (stats['failed_generations'] / stats['total_requests']) * 100
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        # Add router statistics
        stats['router_statistics'] = self.router.get_routing_statistics()
        
        # Add system information
        stats['system_info'] = self.system_info
        
        return stats
    
    def reset_statistics(self):
        """
        Reset all system statistics to initial values.
        
        This method is useful for testing or when you want to start fresh
        with statistics collection.
        """
        if self.enable_statistics:
            self.statistics = {
                'total_requests': 0,
                'successful_generations': 0,
                'failed_generations': 0,
                'languages_used': {},
                'topics_processed': [],
                'average_generation_time': 0.0,
                'system_start_time': datetime.now().isoformat()
            }
        
        # Reset router statistics
        self.router.reset_statistics()
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        
        Returns:
            List[str]: List of supported language names
        """
        return self.system_info['supported_languages'].copy()
    
    def validate_topic(self, topic: str) -> Dict[str, Any]:
        """
        Validate a topic and provide suggestions.
        
        This helper method validates a topic and provides useful information
        about how it would be processed by the system.
        
        Args:
            topic (str): The topic to validate
            
        Returns:
            Dict[str, Any]: Validation results and suggestions
        """
        try:
            # Basic validation
            if not topic or not isinstance(topic, str):
                return {'valid': False, 'error': 'Topic must be a non-empty string'}
            
            if len(topic.strip()) == 0:
                return {'valid': False, 'error': 'Topic cannot be empty or whitespace only'}
            
            if len(topic) > 500:
                return {'valid': False, 'error': 'Topic is too long (max 500 characters)'}
            
            # Classify the topic
            classification_result = self.router.topic_classifier.classify_topic(topic)
            
            # Get writer suggestions
            if classification_result.category == "Tech":
                suggestions = self.router.tech_writer.get_tech_tone_suggestions(topic)
            else:
                suggestions = self.router.general_writer.get_content_suggestions(topic)
            
            return {
                'valid': True,
                'classification': {
                    'category': classification_result.category,
                    'confidence': classification_result.confidence,
                    'reasoning': classification_result.reasoning
                },
                'suggestions': suggestions,
                'estimated_writer': 'Tech Writer Agent' if classification_result.category == "Tech" else 'General Writer Agent'
            }
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation failed: {str(e)}'}
    
    def export_statistics(self, file_path: str, format: str = "json"):
        """
        Export system statistics to a file.
        
        Args:
            file_path (str): Path to save the statistics file
            format (str): Export format ("json" or "csv")
            
        Raises:
            ValueError: If format is not supported
            IOError: If file cannot be written
        """
        if not self.enable_statistics or not self.statistics:
            raise ValueError("Statistics collection is disabled")
        
        stats = self.get_system_statistics()
        
        if format.lower() == "json":
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, default=str)
        elif format.lower() == "csv":
            import csv
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write basic statistics as CSV
                for key, value in stats.items():
                    if isinstance(value, (int, float, str)):
                        writer.writerow([key, value])
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Returns:
            Dict[str, Any]: Detailed system information and capabilities
        """
        return {
            'system_info': self.system_info,
            'configuration': {
                'default_language': self.default_language,
                'statistics_enabled': self.enable_statistics,
                'config_loaded': bool(self.config)
            },
            'capabilities': {
                'topic_classification': True,
                'conditional_routing': True,
                'multi_language_support': True,
                'batch_processing': True,
                'performance_monitoring': self.enable_statistics,
                'export_capabilities': True
            },
            'supported_languages': self.get_supported_languages()
        }
