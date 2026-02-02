"""
General Writer Agent

This module contains the GeneralWriterAgent class responsible for generating
professional LinkedIn posts focused on non-technology topics.
This agent specializes in creating engaging content for business, lifestyle,
leadership, and other general professional topics.
"""

from typing import Dict, Any, Optional
from langchain.schema import BasePromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
import os


class GeneralPostResult(BaseModel):
    """
    Data model for general LinkedIn post results.
    
    Attributes:
        topic (str): The original general topic
        language (str): The language in which the post was written
        post_content (str): The complete LinkedIn post content
        word_count (int): Number of words in the generated post
        paragraph_count (int): Number of paragraphs in the post
        has_call_to_action (bool): Whether the post includes a call-to-action
        content_category (str): Type of content (Business/Lifestyle/Leadership/Education/etc.)
        engagement_type (str): Type of engagement focus (Inspirational/Educational/Discussion)
    """
    topic: str = Field(description="The original general topic for the post")
    language: str = Field(description="The language in which the post was written")
    post_content: str = Field(description="The complete LinkedIn post content")
    word_count: int = Field(description="Number of words in the generated post")
    paragraph_count: int = Field(description="Number of paragraphs in the post")
    has_call_to_action: bool = Field(description="Whether the post includes a call-to-action")
    content_category: str = Field(description="Type of content category")
    engagement_type: str = Field(description="Type of engagement focus")


class GeneralWriterAgent:
    """
    An AI agent specialized in writing general LinkedIn posts.
    
    This agent creates professional, engaging LinkedIn content specifically
    tailored for non-technology topics including business, leadership,
    lifestyle, education, and other general professional subjects.
    
    Attributes:
        llm (ChatOpenAI): The language model for content generation
        writing_chain (LLMChain): The LangChain for generating general posts
        supported_languages (list): List of supported languages for content generation
        content_categories (list): List of general content categories
        engagement_types (list): Types of engagement approaches
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        """
        Initialize the General Writer Agent.
        
        Args:
            model_name (str): Name of the OpenAI model to use. Defaults to "gpt-3.5-turbo"
            temperature (float): Temperature for creative content generation. Higher values for more creativity
        
        Raises:
            ValueError: If model_name is invalid or temperature is out of range
        """
        # Validate inputs
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            raise ValueError("temperature must be a number between 0 and 2")
        
        # Initialize the language model with parameters optimized for creative writing
        self.llm = Ollama(
            model=model_name if model_name != "gpt-3.5-turbo" else "llama3.2:3b",
            temperature=temperature  # Higher temperature for more creative content
        )
        
        # Define the general writing prompt template
        # This prompt is specifically designed for non-technology content
        self.general_writing_prompt = PromptTemplate(
            input_variables=["topic", "language"],
            template="""You are a professional LinkedIn content writer.

Write a LinkedIn post about the given topic in the specified language.

CRITICAL REQUIREMENTS:
1. Write EXACTLY 3 paragraphs. No more, no fewer.
2. Each paragraph must be 2-3 sentences maximum.
3. The final paragraph MUST include a call-to-action question AND hashtags.
4. NO headings, bullet points, or numbering.
5. NO explanations, meta text, or introductory phrases.
6. Start IMMEDIATELY with the first paragraph.
7. Use professional tone.
8. Hashtags go at the VERY END of the final paragraph.

Topic: {topic}
Language: {language}

START WRITING NOW:"""
        )
        
        # Create the writing chain
        self.writing_chain = LLMChain(
            llm=self.llm,
            prompt=self.general_writing_prompt
        )
        
        # Supported languages for content generation
        self.supported_languages = [
            'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese',
            'Dutch', 'Russian', 'Chinese', 'Japanese', 'Korean', 'Arabic',
            'Hindi', 'Bengali', 'Tamil', 'Telugu', 'Marathi', 'Gujarati'
        ]
        
        # General content categories for classification
        self.content_categories = [
            'Business', 'Leadership', 'Workplace', 'Personal Development',
            'Wellness', 'Education', 'Marketing', 'Sales', 'Finance',
            'Human Resources', 'Social Impact', 'Lifestyle', 'Industry',
            'Strategy', 'Innovation', 'Culture', 'Communication'
        ]
        
        # Engagement types for different approaches
        self.engagement_types = [
            'Inspirational', 'Educational', 'Discussion', 'Advice',
            'Storytelling', 'Question-based', 'Call-to-action', 'Reflective'
        ]
    
    def _validate_language(self, language: str) -> str:
        """
        Validate and normalize the language input.
        
        Args:
            language (str): The language to validate
            
        Returns:
            str: Normalized language name
            
        Raises:
            ValueError: If language is not supported
        """
        if not language or not isinstance(language, str):
            raise ValueError("Language must be a non-empty string")
        
        # Normalize language (case-insensitive matching)
        language_normalized = language.strip().title()
        
        if language_normalized not in self.supported_languages:
            raise ValueError(f"Language '{language}' is not supported. Supported languages: {', '.join(self.supported_languages)}")
        
        return language_normalized
    
    def _categorize_content(self, topic: str, content: str) -> tuple[str, str]:
        """
        Categorize the content type and engagement approach.
        
        Args:
            topic (str): The original topic
            content (str): The generated content
            
        Returns:
            tuple[str, str]: Content category and engagement type
        """
        topic_lower = topic.lower()
        content_lower = content.lower()
        
        # Determine content category
        category = 'General'  # Default category
        
        category_keywords = {
            'Business': ['business', 'strategy', 'management', 'operations'],
            'Leadership': ['leadership', 'leader', 'management', 'team'],
            'Workplace': ['workplace', 'office', 'team', 'colleagues'],
            'Personal Development': ['development', 'growth', 'learning', 'skills'],
            'Wellness': ['wellness', 'health', 'balance', 'mental'],
            'Education': ['education', 'learning', 'training', 'knowledge'],
            'Marketing': ['marketing', 'branding', 'promotion', 'customer'],
            'Sales': ['sales', 'selling', 'revenue', 'client'],
            'Finance': ['finance', 'money', 'investment', 'budget'],
            'Human Resources': ['hr', 'hiring', 'recruitment', 'employees'],
            'Social Impact': ['social', 'impact', 'community', 'sustainability'],
            'Lifestyle': ['lifestyle', 'habits', 'routine', 'living']
        }
        
        for cat, keywords in category_keywords.items():
            if any(keyword in topic_lower or keyword in content_lower for keyword in keywords):
                category = cat
                break
        
        # Determine engagement type
        engagement = 'Discussion'  # Default engagement
        
        engagement_keywords = {
            'Inspirational': ['inspire', 'motivation', 'dream', 'passion'],
            'Educational': ['learn', 'educate', 'knowledge', 'insight'],
            'Storytelling': ['story', 'experience', 'journey', 'memory'],
            'Advice': ['advice', 'tip', 'recommendation', 'suggest'],
            'Question-based': ['question', 'ask', 'wonder', 'curious'],
            'Reflective': ['reflect', 'think', 'consider', 'ponder'],
            'Call-to-action': ['action', 'do', 'implement', 'start']
        }
        
        for eng, keywords in engagement_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                engagement = eng
                break
        
        return category, engagement
    
    def _analyze_post_content(self, content: str) -> Dict[str, Any]:
        """
        Analyze the generated post content for quality metrics.
        
        Args:
            content (str): The generated post content
            
        Returns:
            Dict[str, Any]: Analysis results including word count, paragraph count, etc.
        """
        # Basic text analysis
        words = content.split()
        
        # Count paragraphs (simple approach: split by double newlines, ignore hashtag-only lines)
        lines = content.split('\n')
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            if not line:
                # Empty line - end current paragraph
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            elif line.startswith('#'):
                # Hashtag line - end current paragraph if any
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                # Don't add hashtags as a paragraph
            else:
                # Regular text line
                current_paragraph.append(line)
        
        # Add the last paragraph if there is one
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Check for call-to-action indicators
        cta_indicators = ['?', 'What are your thoughts?', 'Share your experience', 
                          'How do you see', 'What\'s your take', 'Comment below',
                          'Let me know', 'Your thoughts?', 'Join the discussion',
                          'What about you?', 'How do you handle']
        has_cta = any(indicator.lower() in content.lower() for indicator in cta_indicators)
        
        return {
            'word_count': len(words),
            'paragraph_count': len(paragraphs),
            'has_call_to_action': has_cta
        }
    
    def _validate_post_requirements(self, content: str, analysis: Dict[str, Any]) -> bool:
        """
        Validate that the generated post meets all requirements.
        
        Args:
            content (str): The generated post content
            analysis (Dict[str, Any]): Analysis results from _analyze_post_content
            
        Returns:
            bool: True if post meets requirements, False otherwise
        """
        # Check paragraph count (2-4 paragraphs required)
        if not (2 <= analysis['paragraph_count'] <= 4):
            return False
        
        # Check word count (150-300 words recommended)
        if not (100 <= analysis['word_count'] <= 400):  # Some flexibility
            return False
        
        # Check for call-to-action
        if not analysis['has_call_to_action']:
            return False
        
        # Check for hashtags (LinkedIn posts typically have hashtags)
        if '#' not in content:
            return False
        
        return True
    
    def generate_general_post(self, topic: str, language: str = "English") -> GeneralPostResult:
        """
        Generate a general-focused LinkedIn post.
        
        This is the main method that creates professional general content
        tailored to the specified topic and language.
        
        Args:
            topic (str): The general topic to write about
            language (str): The language for the post. Defaults to "English"
            
        Returns:
            GeneralPostResult: Object containing the generated post and metadata
            
        Raises:
            ValueError: If topic is invalid or language is not supported
            RuntimeError: If post generation fails or doesn't meet requirements
        """
        # Validate inputs
        if not topic or not isinstance(topic, str):
            raise ValueError("Topic must be a non-empty string")
        
        if len(topic.strip()) == 0:
            raise ValueError("Topic cannot be empty or whitespace only")
        
        # Validate and normalize language
        validated_language = self._validate_language(language)
        
        # Generate the post using the LLM chain
        try:
            response = self.writing_chain.invoke({"topic": topic, "language": validated_language})
            
            # Extract the text content from the response dictionary
            post_content = response.get('text', '').strip() if isinstance(response, dict) else str(response).strip()
            
            # Remove any meta text that might appear at the beginning
            meta_phrases = [
                "Here's a rewritten version of the post that meets the strict requirements:",
                "Here's a revised version of the LinkedIn post that meets the strict requirements:",
                "Here is the post:",
                "This post:",
                "Here's a revised version:",
                "Here's a rewritten version:",
                "Here's a rewritten version of the LinkedIn post that meets the strict requirements:",
                "Here's a rewritten LinkedIn post that meets the strict requirements:",
                "Here's a rewritten LinkedIn post:",
                "Here's a rewritten version:"
            ]
            
            # Remove any meta text from the beginning
            for phrase in meta_phrases:
                if post_content.startswith(phrase):
                    post_content = post_content[len(phrase):].strip()
                    break
            
            # Also remove any meta text that might appear after newlines
            lines = post_content.split('\n')
            if lines and any(lines[0].startswith(phrase) for phrase in meta_phrases):
                post_content = '\n'.join(lines[1:]).strip()
            
            # Analyze the generated content
            analysis = self._analyze_post_content(post_content)
            
            # Validate that the post meets requirements
            if not self._validate_post_requirements(post_content, analysis):
                # If requirements aren't met, try once more with a more specific prompt
                post_content = self._regenerate_post_with_requirements(topic, validated_language, post_content)
                analysis = self._analyze_post_content(post_content)
            
            # Categorize the content
            content_category, engagement_type = self._categorize_content(topic, post_content)
            
            return GeneralPostResult(
                topic=topic,
                language=validated_language,
                post_content=post_content,
                word_count=analysis['word_count'],
                paragraph_count=analysis['paragraph_count'],
                has_call_to_action=analysis['has_call_to_action'],
                content_category=content_category,
                engagement_type=engagement_type
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate general post: {str(e)}")
    
    def _regenerate_post_with_requirements(self, topic: str, language: str, previous_content: str) -> str:
        """
        Regenerate a post when the first attempt doesn't meet requirements.
        
        This is a fallback method that uses a more specific prompt to ensure
        all requirements are met.
        
        Args:
            topic (str): The general topic
            language (str): The target language
            previous_content (str): The previously generated content for reference
            
        Returns:
            str: The regenerated post content
        """
        regeneration_prompt = PromptTemplate(
            input_variables=["topic", "language", "previous_content"],
            template="""The previous attempt didn't meet all requirements. Please regenerate the LinkedIn post with these strict requirements:

Topic: "{topic}"
Language: "{language}"

STRICT REQUIREMENTS:
1. Exactly 2-4 paragraphs (no more, no less)
2. 150-300 words total
3. Must end with a question or call-to-action
4. Must include 3-5 relevant hashtags
5. Professional but approachable tone
6. Focus on human insights and experiences

Previous attempt for reference:
"{previous_content}"

Please write the improved LinkedIn post now:"""
        )
        
        regeneration_chain = LLMChain(
            llm=self.llm,
            prompt=regeneration_prompt
        )
        
        response = regeneration_chain.invoke({"topic": topic, "language": language, "previous_content": previous_content})
        return response.get('text', '').strip() if isinstance(response, dict) else str(response).strip()
    
    def generate_multiple_posts(self, topics: list, language: str = "English") -> list[GeneralPostResult]:
        """
        Generate multiple general posts for different topics.
        
        This method processes multiple topics efficiently, which is useful
        for content creation campaigns or testing.
        
        Args:
            topics (list): List of general topics
            language (str): The language for all posts. Defaults to "English"
            
        Returns:
            list[GeneralPostResult]: List of generated post results
            
        Raises:
            ValueError: If topics list is empty or contains invalid items
        """
        if not topics or not isinstance(topics, list):
            raise ValueError("Topics must be a non-empty list")
        
        results = []
        
        for topic in topics:
            try:
                result = self.generate_general_post(topic, language)
                results.append(result)
            except Exception as e:
                # Create an error result for failed generations
                error_result = GeneralPostResult(
                    topic=topic,
                    language=language,
                    post_content=f"Error generating post: {str(e)}",
                    word_count=0,
                    paragraph_count=0,
                    has_call_to_action=False,
                    content_category="Error",
                    engagement_type="Error"
                )
                results.append(error_result)
        
        return results
    
    def get_content_suggestions(self, topic: str) -> list[str]:
        """
        Get content suggestions for a given general topic.
        
        This helper method provides recommendations on how to approach
        writing about specific general topics.
        
        Args:
            topic (str): The general topic to analyze
            
        Returns:
            list[str]: List of content suggestions
        """
        topic_lower = topic.lower()
        suggestions = []
        
        # Analyze topic characteristics and provide suggestions
        if any(keyword in topic_lower for keyword in ['leadership', 'leader', 'management']):
            suggestions.append("Share personal leadership experiences or lessons")
            suggestions.append("Include examples of successful leadership strategies")
        
        if any(keyword in topic_lower for keyword in ['workplace', 'culture', 'team']):
            suggestions.append("Focus on creating positive work environments")
            suggestions.append("Include team collaboration and communication tips")
        
        if any(keyword in topic_lower for keyword in ['balance', 'wellness', 'health']):
            suggestions.append("Emphasize practical self-care strategies")
            suggestions.append("Share personal wellness journey or tips")
        
        if any(keyword in topic_lower for keyword in ['learning', 'education', 'skills']):
            suggestions.append("Focus on continuous learning and development")
            suggestions.append("Include specific skill-building recommendations")
        
        if any(keyword in topic_lower for keyword in ['business', 'strategy', 'growth']):
            suggestions.append("Highlight strategic thinking and planning")
            suggestions.append("Include business growth insights or case studies")
        
        if any(keyword in topic_lower for keyword in ['communication', 'networking', 'relationships']):
            suggestions.append("Focus on building professional relationships")
            suggestions.append("Include effective communication strategies")
        
        if not suggestions:
            suggestions.append("Share personal experiences and insights")
            suggestions.append("Focus on practical advice and takeaways")
        
        return suggestions
