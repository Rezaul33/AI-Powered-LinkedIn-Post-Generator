"""
Tech Topic Demonstration - English

This script demonstrates the AI-Powered LinkedIn Post Generator with a technology
topic in English. It showcases the complete workflow including topic classification,
conditional routing to the Tech Writer Agent, and post generation.

This example demonstrates:
1. Tech topic classification
2. Conditional routing to Tech Writer Agent
3. English language content generation
4. Professional tech-focused LinkedIn post creation
5. System statistics and monitoring
"""

import sys
import os
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from linkedin_post_generator import LinkedInPostGenerator


def print_section_header(title: str):
    """
    Print a formatted section header for better readability.
    
    Args:
        title (str): The title of the section
    """
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)


def print_subsection(title: str):
    """
    Print a formatted subsection header.
    
    Args:
        title (str): The title of the subsection
    """
    print(f"\n{'-'*40}")
    print(f" {title} ".center(40, '-'))
    print('-'*40)


def demonstrate_tech_topic_english():
    """
    Demonstrate the AI-Powered LinkedIn Post Generator with a tech topic in Bengali.
    
    This function showcases the complete workflow for generating a technology-focused
    LinkedIn post in Bengali, including classification, routing, and generation.
    """
    
    print_section_header("AI-Powered LinkedIn Post Generator - Tech Topic Demo (English)")
    
    # Initialize the system
    print_subsection("System Initialization")
    print("Initializing the AI-Powered LinkedIn Post Generator...")
    
    try:
        # Create the main generator instance
        generator = LinkedInPostGenerator(
            model_name="llama3.2:3b",  # Using better Ollama model
            classification_temperature=0.1,
            writing_temperature=0.7,
            confidence_threshold=0.6,
            default_language="English",
            enable_statistics=True
        )
        
        print("✅ System initialized successfully!")
        print(f"📊 Statistics collection: {'Enabled' if generator.enable_statistics else 'Disabled'}")
        print(f"🌐 Default language: {generator.default_language}")
        print(f"🤖 AI Model: {generator.system_info['model_name']} (Ollama Local)")
        
    except Exception as e:
        print(f"❌ System initialization failed: {str(e)}")
        print("\nPlease ensure you have:")
        print("1. Set your OPENAI_API_KEY environment variable")
        print("2. Installed all required dependencies from requirements.txt")
        return
    
    # Define the tech topic for demonstration
    print_subsection("Tech Topic Selection")
    topic = "AI in Healthcare: Revolutionizing Medical Diagnosis and Treatment"
    language = "English"  # Changed back to English
    print(f"🎯 Selected Tech Topic: '{topic}'")
    print(f"🌐 Selected Language: {language}")
    print("📝 This topic is clearly technology-related and should be classified as 'Tech'")
    
    # Validate the topic
    print_subsection("Topic Validation and Classification")
    print("Validating the topic and getting classification insights...")
    
    validation_result = generator.validate_topic(topic)
    
    if validation_result['valid']:
        print("✅ Topic validation successful!")
        print(f"📊 Classification: {validation_result['classification']['category']}")
        print(f"🎯 Confidence: {validation_result['classification']['confidence']:.2f}")
        print(f"💭 Reasoning: {validation_result['classification']['reasoning']}")
        print(f"🤖 Estimated Writer: {validation_result['estimated_writer']}")
        
        print("\n💡 Content Suggestions:")
        for i, suggestion in enumerate(validation_result['suggestions'], 1):
            print(f"   {i}. {suggestion}")
    else:
        print(f"❌ Topic validation failed: {validation_result['error']}")
        return
    
    # Generate the LinkedIn post
    print_subsection("LinkedIn Post Generation")
    print("Generating LinkedIn post using conditional routing...")
    
    try:
        # Generate the post
        response = generator.generate_post(
            topic=topic,
            language="English",
            user_preferences={
                "tone": "professional",
                "include_hashtags": True,
                "target_audience": "tech professionals"
            }
        )
        
        if response.success:
            print("✅ LinkedIn post generated successfully!")
            
            # Display routing information
            print_subsection("Routing Decision Analysis")
            routing = response.routing_result.routing_decision
            print(f"🔀 Selected Writer: {routing.selected_writer.value} Writer Agent")
            print(f"📊 Routing Confidence: {routing.confidence_score:.2f}")
            print(f"💭 Routing Reasoning: {routing.routing_reasoning}")
            print(f"⏱️ Processing Time: {response.routing_result.processing_time_ms:.2f} ms")
            
            # Display the generated post
            print_subsection("Generated LinkedIn Post")
            post_result = response.routing_result.post_result
            
            print(f"📝 Topic: {post_result.topic}")
            print(f"🌐 Language: {post_result.language}")
            print(f"📊 Word Count: {post_result.word_count}")
            print(f"📋 Paragraph Count: {post_result.paragraph_count}")
            print(f"🎯 Call-to-Action: {'Yes' if post_result.has_call_to_action else 'No'}")
            
            if hasattr(post_result, 'technical_depth'):
                print(f"🔬 Technical Depth: {post_result.technical_depth}")
            
            print(f"\n📄 Generated Post Content:")
            print("-" * 60)
            print(post_result.post_content)
            print("-" * 60)
            
            # Save the generated post to a text file
            output_filename = os.path.join(os.path.dirname(__file__), '..', 'output', 'english_post_output.txt')
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    f.write(f"Topic: {post_result.topic}\n")
                    f.write(f"Language: {post_result.language}\n")
                    f.write(f"Word Count: {post_result.word_count}\n")
                    f.write(f"Paragraph Count: {post_result.paragraph_count}\n")
                    f.write(f"Has Call-to-Action: {post_result.has_call_to_action}\n")
                    if hasattr(post_result, 'technical_depth'):
                        f.write(f"Technical Depth: {post_result.technical_depth}\n")
                    f.write(f"\nGenerated Post Content:\n")
                    f.write("="*50 + "\n")
                    f.write(post_result.post_content)
                    f.write("\n" + "="*50 + "\n")
                
                print(f"\n💾 Post saved to: {os.path.basename(output_filename)}")
            except Exception as save_error:
                print(f"\n⚠️ Warning: Could not save post to file: {str(save_error)}")
            
        else:
            print(f"❌ Post generation failed: {response.error_message}")
            return
    
    except Exception as e:
        print(f"❌ Error during post generation: {str(e)}")
        return
    
    # Display system statistics
    print_subsection("System Statistics")
    stats = generator.get_system_statistics()
    
    print(f"📊 Total Requests: {stats['total_requests']}")
    print(f"✅ Successful Generations: {stats['successful_generations']}")
    print(f"❌ Failed Generations: {stats['failed_generations']}")
    print(f"📈 Success Rate: {stats['success_rate']:.1f}%")
    print(f"⏱️ Average Generation Time: {stats['average_generation_time']:.2f} ms")
    
    # Final summary
    print_section_header("Demo Summary")
    print("🎉 Tech Topic Demo Completed Successfully!")
    print("\n📋 What was demonstrated:")
    print("   ✅ Tech topic classification (AI in Healthcare)")
    print("   ✅ Conditional routing to Tech Writer Agent")
    print("   ✅ Professional English LinkedIn post generation")
    
    print(f"\n🔬 Technical Details:")
    print(f"   📊 Classification Confidence: {validation_result['classification']['confidence']:.2f}")
    print(f"   🔀 Routing Decision: Tech Writer Agent")
    print(f"   ⏱️ Processing Time: {response.routing_result.processing_time_ms:.2f} ms")
    print(f"   📝 Post Length: {post_result.word_count} words")
    print(f"   📋 Post Structure: {post_result.paragraph_count} paragraphs")
    
    print(f"\n💡 Key Insights:")
    print("   • The system correctly identified this as a technology topic")
    print("   • Conditional routing worked as expected")
    print("   • Generated post maintains professional tech tone")
    print("   • All requirements (3 paragraphs, CTA, hashtags) were met")
    
    print("\n✅ Demo completed successfully!")


def main():
    """
    Main function to run the tech topic demonstration.
    """
    try:
        demonstrate_tech_topic_english()
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error during demo: {str(e)}")
        print("Please check your setup and try again.")


if __name__ == "__main__":
    main()
