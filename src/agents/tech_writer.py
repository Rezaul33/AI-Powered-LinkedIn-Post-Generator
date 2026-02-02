"""
Tech Writer Agent

This module contains the TechWriterAgent class responsible for generating
professional LinkedIn posts focused on technology-related topics.
This agent specializes in creating engaging, informative tech content
with appropriate technical depth and professional tone.
"""

from typing import Dict, Any, Optional
from langchain.schema import BasePromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
import os


class TechPostResult(BaseModel):
    """
    Data model for technology-focused LinkedIn post results.
    
    Attributes:
        topic (str): The original technology topic
        language (str): The language in which the post was written
        post_content (str): The complete LinkedIn post content
        word_count (int): Number of words in the generated post
        paragraph_count (int): Number of paragraphs in the post
        has_call_to_action (bool): Whether the post includes a call-to-action
        technical_depth (str): Level of technical complexity (Basic/Intermediate/Advanced)
    """
    topic: str = Field(description="The original technology topic for the post")
    language: str = Field(description="The language in which the post was written")
    post_content: str = Field(description="The complete LinkedIn post content")
    word_count: int = Field(description="Number of words in the generated post")
    paragraph_count: int = Field(description="Number of paragraphs in the post")
    has_call_to_action: bool = Field(description="Whether the post includes a call-to-action")
    technical_depth: str = Field(description="Level of technical complexity")


class TechWriterAgent:
    """
    An AI agent specialized in writing technology-focused LinkedIn posts.
    
    This agent creates professional, engaging LinkedIn content specifically
    tailored for technology topics. It understands technical concepts,
    industry trends, and the appropriate tone for tech professionals on LinkedIn.
    
    Attributes:
        llm (ChatOpenAI): The language model for content generation
        writing_chain (LLMChain): The LangChain for generating tech posts
        supported_languages (list): List of supported languages for content generation
        tech_tone_keywords (list): Keywords that define the professional tech tone
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        """
        Initialize the Tech Writer Agent.
        
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
        
        # Define the tech writing prompt template
        # This prompt is specifically designed for technology content
        self.tech_writing_prompt = PromptTemplate(
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
            prompt=self.tech_writing_prompt
        )
        
        # Supported languages for content generation
        self.supported_languages = [
            'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese',
            'Dutch', 'Russian', 'Chinese', 'Japanese', 'Korean', 'Arabic',
            'Hindi', 'Bengali', 'Tamil', 'Telugu', 'Marathi', 'Gujarati'
        ]
        
        # Keywords that define professional tech writing tone
        self.tech_tone_keywords = [
            'innovative', 'cutting-edge', 'breakthrough', 'revolutionary',
            'transformative', 'scalable', 'efficient', 'optimized',
            'robust', 'secure', 'intelligent', 'automated', 'seamless'
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
    
    def _analyze_post_content(self, post_content: str) -> Dict[str, Any]:
        """
        Analyze the generated post content to extract key metrics.
        
        Args:
            post_content (str): The generated LinkedIn post content
            
        Returns:
            Dict[str, Any]: Analysis results including word count, paragraph count, etc.
        """
        # Count words
        word_count = len(post_content.split())
        
        # Count paragraphs (simple approach: split by double newlines, ignore hashtag-only lines)
        lines = post_content.split('\n')
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
        
        paragraph_count = len(paragraphs)
        
        # Check for call-to-action
        has_call_to_action = any(
            '?' in para or 
            any(word in para.lower() for word in 
            ['what do you think', 'share your', 'let me know', 'your thoughts', 'comment below', 'join the discussion', 'what about you', 'how do you', 'we need', 'we must'])
            for para in paragraphs
        )
        
        # Check for hashtags
        has_hashtags = '#' in post_content
        
        # Calculate technical depth based on content
        tech_keywords = ['ai', 'machine learning', 'algorithm', 'data', 'technology', 'software', 'programming', 'code', 'digital', 'automation']
        tech_depth_score = sum(1 for keyword in tech_keywords if keyword.lower() in post_content.lower())
        
        if tech_depth_score >= 3:
            technical_depth = "Advanced"
        elif tech_depth_score >= 1:
            technical_depth = "Intermediate"
        else:
            technical_depth = "Basic"
        
        return {
            'word_count': word_count,
            'paragraph_count': paragraph_count,
            'has_call_to_action': has_call_to_action,
            'has_hashtags': has_hashtags,
            'technical_depth': technical_depth,
            'line_count': len(post_content.split('\n')),
            'character_count': len(post_content)
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
    
    def generate_tech_post(self, topic: str, language: str = "English") -> TechPostResult:
        """
        Generate a technology-focused LinkedIn post.
        
        This is the main method that creates professional tech content
        tailored to the specified topic and language.
        
        Args:
            topic (str): The technology topic to write about
            language (str): The language for the post. Defaults to "English"
            
        Returns:
            TechPostResult: Object containing the generated post and metadata
            
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
            
            # Force compliance: if still not 3 paragraphs, truncate or pad
            if analysis['paragraph_count'] != 3:
                post_content = self._force_paragraph_compliance(post_content, analysis['paragraph_count'])
                analysis = self._analyze_post_content(post_content)
            
            return TechPostResult(
                topic=topic,
                language=validated_language,
                post_content=post_content,
                word_count=analysis['word_count'],
                paragraph_count=analysis['paragraph_count'],
                has_call_to_action=analysis['has_call_to_action'],
                technical_depth=analysis['technical_depth']
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate tech post: {str(e)}")
    
    def _regenerate_post_with_requirements(self, topic: str, language: str, previous_content: str) -> str:
        """
        Regenerate a post when the first attempt doesn't meet requirements.
        
        This is a fallback method that uses a more specific prompt to ensure
        all requirements are met.
        
        Args:
            topic (str): The technology topic
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
5. Professional tech tone

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
    
    def _force_paragraph_compliance(self, content: str, current_count: int) -> str:
        """
        Force the content to have exactly 3 paragraphs by truncating or padding.
        
        Args:
            content (str): The generated content
            current_count (int): Current paragraph count
            
        Returns:
            str: Content with exactly 3 paragraphs
        """
        lines = content.split('\n')
        paragraphs = []
        current_paragraph = []
        
        # Parse paragraphs
        for line in lines:
            line = line.strip()
            if not line:
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            elif line.startswith('#'):
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            else:
                current_paragraph.append(line)
        
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Force exactly 3 paragraphs
        if len(paragraphs) > 3:
            # Truncate to first 3 paragraphs
            paragraphs = paragraphs[:3]
        elif len(paragraphs) < 3:
            # Pad with generic content
            while len(paragraphs) < 3:
                if len(paragraphs) == 1:
                    paragraphs.append("This technology continues to evolve and shape our future in remarkable ways.")
                else:
                    paragraphs.append("What are your thoughts on these developments? #Innovation #Technology")
        
        # Reconstruct content
        result = '\n\n'.join(paragraphs)
        return result
    
    def generate_multiple_posts(self, topics: list, language: str = "English") -> list[TechPostResult]:
        """
        Generate multiple tech posts for different topics.
        
        This method processes multiple topics efficiently, which is useful
        for content creation campaigns or testing.
        
        Args:
            topics (list): List of technology topics
            language (str): The language for all posts. Defaults to "English"
            
        Returns:
            list[TechPostResult]: List of generated post results
            
        Raises:
            ValueError: If topics list is empty or contains invalid items
        """
        if not topics or not isinstance(topics, list):
            raise ValueError("Topics must be a non-empty list")
        
        results = []
        
        for topic in topics:
            try:
                result = self.generate_tech_post(topic, language)
                results.append(result)
            except Exception as e:
                # Create an error result for failed generations
                error_result = TechPostResult(
                    topic=topic,
                    language=language,
                    post_content=f"Error generating post: {str(e)}",
                    word_count=0,
                    paragraph_count=0,
                    has_call_to_action=False,
                    technical_depth="Unknown"
                )
                results.append(error_result)
        
        return results
    
    def get_tech_tone_suggestions(self, topic: str) -> list[str]:
        """
        Get tone and style suggestions for a given tech topic.
        
        This helper method provides recommendations on how to approach
        writing about specific technology topics.
        
        Args:
            topic (str): The technology topic to analyze
            
        Returns:
            list[str]: List of tone and style suggestions
        """
        topic_lower = topic.lower()
        suggestions = []
        
        # Analyze topic characteristics and provide suggestions
        if any(keyword in topic_lower for keyword in ['ai', 'machine learning', 'deep learning']):
            suggestions.append("Focus on practical applications and ethical considerations")
            suggestions.append("Include recent breakthroughs or research findings")
        
        if any(keyword in topic_lower for keyword in ['cybersecurity', 'security', 'privacy']):
            suggestions.append("Emphasize importance and urgency")
            suggestions.append("Include actionable security tips")
        
        if any(keyword in topic_lower for keyword in ['cloud', 'aws', 'azure', 'gcp']):
            suggestions.append("Highlight cost benefits and scalability")
            suggestions.append("Include migration strategies or best practices")
        
        if any(keyword in topic_lower for keyword in ['blockchain', 'cryptocurrency']):
            suggestions.append("Explain complex concepts simply")
            suggestions.append("Focus on real-world use cases beyond speculation")
        
        if not suggestions:
            suggestions.append("Focus on business impact and ROI")
            suggestions.append("Include future trends and predictions")
        
        return suggestions
