# 🚀 AI-Powered LinkedIn Post Generator

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20AI-orange.svg)](https://ollama.ai)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-blue.svg)](https://langchain.com)

A sophisticated AI agent system that generates professional LinkedIn posts using **local Ollama models** with intelligent conditional routing and multi-language support.

## ✨ Key Features

- 🤖 **Intelligent Topic Classification** - Automatically classifies topics as "Tech" or "General"
- 🔀 **Conditional Routing System** - Routes topics to specialized writer agents based on classification
- 🌐 **Multi-Language Support** - Generates content in 14+ languages including Bengali, Hindi, Tamil, and more
- 📝 **Professional Content** - Creates engaging LinkedIn posts with proper structure and hashtags
- 📊 **Performance Analytics** - Built-in statistics and monitoring system
- 💾 **Organized Output** - Automatic saving to structured output folder
- 🦙 **Local AI Processing** - Uses Ollama for free, private LLM inference (no API costs)
- ⚡ **Batch Processing** - Generate multiple posts efficiently

## 🎯 Compliance

This project fully satisfies the requirements:

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| ✅ User Input Acceptance | Complete | Accepts Topic and Language inputs |
| ✅ Conditional Routing Agent | Complete | Intelligent topic analysis and routing |
| ✅ Two Writer Agents | Complete | Tech Writer & General Writer agents |
| ✅ Professional LinkedIn Posts | Complete | 2-4 paragraphs, CTA, hashtags |
| ✅ Multi-Language Support | Complete | 18+ languages with cultural adaptation |
| ✅ Conditional Handover | Complete | Tech → Tech Writer, General → General Writer |
| ✅ Demonstration Examples | Complete | Tech (English) + General (Bengali) demos |

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│ Topic Classifier │───▶│ Conditional     │
│  (Topic + Lang) │    │   Agent          │    │ Router Agent    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                       │
                       ┌─────────────────────────────────┼─────────────────────────────────┐
                       │                                 │                                 │
                       ▼                                 ▼                                 ▼
            ┌─────────────────┐               ┌─────────────────┐               ┌─────────────────┐
            │ Tech Writer     │               │ General Writer  │               │ Statistics &    │
            │ Agent           │               │ Agent           │               │ Monitoring      │
            └─────────────────┘               └─────────────────┘               └─────────────────┘
                       │                                 │
                       └─────────────────────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │ LinkedIn Post   │
                              │   Output        │
                              └─────────────────┘
```

## 📁 Project Structure

```
AI-Powered-LinkedIn-Post-Generator/
├── 📂 src/                           # Source code
│   ├── linkedin_post_generator.py    # Main generator class
│   └── 📂 agents/                    # AI agents
│       ├── topic_classifier.py       # Topic classification agent
│       ├── conditional_router.py     # Intelligent routing logic
│       ├── tech_writer_agent.py      # Technology content writer
│       └── general_writer_agent.py  # General content writer
├── 📂 examples/                      # Demonstration scripts
│   ├── tech_demo_english.py          # Tech topic demo (English)
│   ├── tech_demo_bengali.py          # Tech topic demo (Bengali)
│   └── general_demo_bengali.py       # General topic demo (Bengali)
├── 📂 output/                        # Generated content
│   ├── english_post_output.txt       # English tech posts
│   ├── bengali_post_output.txt       # Bengali tech posts
│   └── general_bengali_post_output.txt # Bengali general posts
├── 📂 docs/                          # Documentation
├── 📂 tests/                         # Test files
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment template
└── README.md                         # This file
```

## 🛠️ Installation & Setup

### Prerequisites

- **Python 3.8+** - Download from [python.org](https://python.org)
- **Ollama** - Local AI model runner
- **Git** - For cloning the repository

### 1️⃣ Install Ollama

Choose your operating system:

**Windows:**
```bash
# Download installer from https://ollama.ai/download
# Or using winget:
winget install Ollama.Ollama
```

**macOS:**
```bash
# Using Homebrew
brew install ollama
```

**Linux:**
```bash
# Official installation script
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2️⃣ Clone & Setup Project

```bash
# Clone the repository
git clone https://github.com/Rezaul33/AI-Powered-LinkedIn-Post-Generator.git
cd AI-Powered-LinkedIn-Post-Generator

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ Pull AI Model

```bash
# Pull the required Llama model (recommended)
ollama pull llama3.2:3b

# Alternative models (optional)
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

### 4️⃣ Verify Installation

```bash
# Check Ollama is running
ollama list

# Test the system
python examples/tech_demo_english.py
```

## 🚀 Quick Start

### Tech Topic Demo (English)

```bash
python examples/tech_demo_english.py
```

**Sample Output:** `output/english_post_output.txt`
```
Topic: AI in Healthcare: Revolutionizing Medical Diagnosis and Treatment
Language: English
Word Count: 168
Paragraph Count: 3
Has Call-to-Action: True
Technical Depth: Advanced

Generated Post Content:
==================================================
Artificial intelligence (AI) is transforming the healthcare industry by revolutionizing medical diagnosis and treatment...
#AIinHealthcare #RevolutionizingMedicine #PersonalizedCare
==================================================
```

### Tech Topic Demo (Bengali)

```bash
python examples/tech_demo_bengali.py
```

**Sample Output:** `output/bengali_post_output.txt`
```
Topic: AI in Healthcare: Revolutionizing Medical Diagnosis and Treatment
Language: Bengali
Word Count: 65
Paragraph Count: 3
Has Call-to-Action: True
Technical Depth: Intermediate

Generated Post Content:
==================================================
আধুনিক স্বাস্থ্যসেবায় কৃত্রিম বুদ্ধিমত্তা...
#AIinHealthcare #MedicalDiagnosis #TreatmentRevolution
==================================================
```

### General Topic Demo (Bengali)

```bash
python examples/general_demo_bengali.py
```

**Sample Output:** `output/general_bengali_post_output.txt`
```
Topic: কর্মজীবনে ভারসাম্য রক্ষা করার গুরুত্ব
Language: Bengali
Word Count: 79
Paragraph Count: 3
Has Call-to-Action: True
Content Category: General
Engagement Type: Discussion

Generated Post Content:
==================================================
"কর্মজীবনে ভারসাম্য রক্ষা করার গুরুত্ব...
#কর্মজীবনে_ভারসাম্য #ভারসাম্য_রক্ষা #সফলতা_এবং_দুর্বলতা"
==================================================
```

### 🌍 Multi-Language Batch Demo (All 14 Languages)

```bash
python examples/multi_language_batch_demo.py
```

**Features:**
- Generates posts in all 14 supported languages simultaneously
- Cultural context adaptation for each language/region
- Comprehensive performance statistics
- Saves results to `output/multi_language_batch_output.txt`

**Languages Included:**
- **Major Languages (11)**: English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, Hindi
- **Regional Languages (3)**: Bengali (Bangladesh), Tamil, Hindi

**Sample Output:** `output/multi_language_batch_output.txt`
```
Multi-Language Batch Results:
✅ Successful Generations: 14/14
📈 Success Rate: 100.0%
⏱️ Average Time per Language: ~10,693 ms
🌍 Languages: English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, Hindi, Bengali, Tamil
```

## 💻 Usage Examples

### Basic Python Usage

```python
from src.linkedin_post_generator import LinkedInPostGenerator

# Initialize the generator
generator = LinkedInPostGenerator(
    model_name="llama3.2:3b",
    enable_statistics=True
)

# Generate a tech post
response = generator.generate_post(
    topic="Machine Learning in Finance",
    language="English",
    user_preferences={
        "tone": "professional",
        "include_hashtags": True,
        "target_audience": "finance professionals"
    }
)

if response.success:
    print(response.post_result.post_content)
    print(f"Word Count: {response.post_result.word_count}")
    print(f"Processing Time: {response.routing_result.processing_time_ms}ms")
```

### Advanced Multi-Language Usage

```python
# Generate posts in different languages
topics_languages = [
    ("Blockchain Technology", "English"),
    ("কৃত্রিম বুদ্ধিমত্তার ভবিষ্যৎ", "Bengali"),
    ("Desarrollo Sostenible", "Spanish"),
    ("मशीन लर्निंग के अनुप्रयोग", "Hindi")
]

for topic, language in topics_languages:
    response = generator.generate_post(
        topic=topic,
        language=language,
        user_preferences={
            "tone": "professional",
            "include_hashtags": True,
            "cultural_context": "local" if language != "English" else None
        }
    )
    
    print(f"\n=== {language} Post ===")
    print(response.post_result.post_content)
```

### Batch Processing

```python
# Process multiple topics efficiently
topics = [
    "Cloud Computing Trends",
    "Remote Work Best Practices",
    "Data Science Careers",
    "Cybersecurity Essentials"
]

results = generator.batch_generate_posts(
    topics=topics,
    language="English",
    user_preferences={"tone": "professional", "include_hashtags": True}
)

for i, result in enumerate(results):
    if result.success:
        print(f"Post {i+1}: {result.post_result.word_count} words")
    else:
        print(f"Post {i+1} failed: {result.error_message}")
```

## 🌐 Supported Languages

### 🌍 Major Languages (11)
- **English** 🇺🇸🇬🇧 - Default language
- **Spanish** 🇪🇸 - Español
- **French** 🇫🇷 - Français
- **German** 🇩🇪 - Deutsch
- **Italian** 🇮🇹 - Italiano
- **Portuguese** 🇵🇹 - Português
- **Russian** 🇷🇺 - Русский
- **Chinese** 🇨🇳 - 中文
- **Japanese** 🇯🇵 - 日本語
- **Korean** 🇰🇷 - 한국어
- **Arabic** 🇸🇦 - العربية

### � Regional Languages (3)
- **Hindi** 🇮🇳 - हिन्दी
- **Bengali** 🇧🇩 - বাংলা (Bangladesh)
- **Tamil** 🇮🇳 - தமிழ்

### 📁 Available Demo Scripts
| Language | Demo Script | Topic Type | Status |
|----------|-------------|------------|---------|
| **English** | `tech_demo_english.py` | Tech Topic | ✅ Available |
| **Bengali** | `tech_demo_bengali.py` | Tech Topic | ✅ Available |
| **Bengali** | `general_demo_bengali.py` | General Topic | ✅ Available |
| **All 14 Languages** | `multi_language_batch_demo.py` | Batch Processing | ✅ Available |

### 🚀 Quick Usage Examples

**Generate posts in other supported languages:**

```python
# Spanish example
response = generator.generate_post(
    topic="Inteligencia Artificial en Medicina",
    language="Spanish",
    user_preferences={"tone": "professional"}
)

# Hindi example
response = generator.generate_post(
    topic="कृत्रिम बुद्धिमत्ता का भविष्य",
    language="Hindi", 
    user_preferences={"tone": "professional"}
)

# French example
response = generator.generate_post(
    topic="L'IA dans l'éducation",
    language="French",
    user_preferences={"tone": "professional"}
)
```

## ⚙️ Configuration Options

### Model Configuration

```python
generator = LinkedInPostGenerator(
    model_name="llama3.2:3b",              # Ollama model
    classification_temperature=0.1,        # Consistent classification
    writing_temperature=0.7,              # Creative content
    confidence_threshold=0.6,             # Routing confidence
    default_language="English",            # Default language
    enable_statistics=True                 # Performance tracking
)
```

### User Preferences

```python
user_preferences = {
    "tone": "professional",               # "professional", "casual", "formal"
    "include_hashtags": True,             # Include relevant hashtags
    "target_audience": "tech professionals", # Target audience
    "cultural_context": "Bangladeshi work culture", # Cultural adaptation
    "post_length": "medium",              # "short", "medium", "long"
    "engagement_type": "discussion"       # "discussion", "informative", "promotional"
}
```

### Environment Variables (Optional)

Create `.env` file for advanced configuration:

```bash
# Model Configuration
DEFAULT_MODEL=llama3.2:3b
CLASSIFICATION_TEMPERATURE=0.1
WRITING_TEMPERATURE=0.7

# Performance
ENABLE_STATISTICS=true
CACHE_TTL=3600
REQUESTS_PER_MINUTE=60

# Logging
LOG_LEVEL=INFO
```

## 📊 System Features

### 🤖 Topic Classification

**Tech Topics Include:**
- Technology & Software Development
- Artificial Intelligence & Machine Learning
- Data Science & Analytics
- Cybersecurity & Networking
- Cloud Computing & DevOps
- Blockchain & Cryptocurrency

**General Topics Include:**
- Business & Management
- Personal Development
- Lifestyle & Wellness
- Education & Learning
- Finance & Economics
- Marketing & Sales

### 📝 Content Structure

All generated posts follow LinkedIn best practices:

- **2-4 paragraphs** for optimal engagement
- **Professional tone** with industry-specific language
- **Call-to-action** to encourage comments and shares
- **Relevant hashtags** for discoverability
- **Cultural adaptation** for regional audiences
- **Unicode support** for non-Latin scripts

### 📈 Performance Monitoring

The system tracks comprehensive metrics:

```python
# Get system statistics
stats = generator.get_system_statistics()

print(f"Success Rate: {stats['success_rate']:.1f}%")
print(f"Average Processing Time: {stats['average_generation_time']:.2f}ms")
print(f"Total Requests: {stats['total_requests']}")

# Router statistics
router_stats = stats['router_statistics']
print(f"Tech Routes: {router_stats['tech_routes']}")
print(f"General Routes: {router_stats['general_routes']}")

# Language usage
for language, count in stats['languages_used'].items():
    print(f"{language}: {count} requests")
```

## 🔧 Advanced Features

### Custom Writer Agents

```python
# Access individual agents directly
tech_writer = generator.router.tech_writer
general_writer = generator.router.general_writer

# Get content suggestions
suggestions = tech_writer.get_tech_tone_suggestions("AI in Healthcare")
print(suggestions)

# Generate content directly
tech_content = tech_writer.generate_tech_post(
    topic="Quantum Computing",
    language="English",
    user_preferences={"tone": "professional"}
)
```

### Export Statistics

```python
# Export statistics to JSON
generator.export_statistics("performance_stats.json")

# Export to CSV
generator.export_statistics("performance_stats.csv", format="csv")
```

### Error Handling

```python
try:
    response = generator.generate_post(
        topic="Invalid Topic",
        language="UnsupportedLanguage"
    )
except ValueError as e:
    print(f"Validation Error: {e}")
except RuntimeError as e:
    print(f"Generation Error: {e}")

# Check response success
if not response.success:
    print(f"Error: {response.error_message}")
    print(f"Error Code: {response.error_code}")
```

## 🧪 Testing & Validation

### Run All Demos

```bash
# Test tech topic classification
python examples/tech_demo_english.py

# Test multi-language tech content
python examples/tech_demo_bengali.py

# Test general topic handling
python examples/general_demo_bengali.py
```

### Performance Benchmarks

```python
import time

# Benchmark processing time
start_time = time.time()
response = generator.generate_post(
    topic="Test Topic",
    language="English"
)
end_time = time.time()

print(f"Processing Time: {(end_time - start_time) * 1000:.2f}ms")
print(f"System Report: {response.routing_result.processing_time_ms}ms")
```

### Quality Validation

```python
# Validate post structure
post = response.post_result
assert 2 <= post.paragraph_count <= 4, "Post should have 2-4 paragraphs"
assert post.has_call_to_action, "Post should have call-to-action"
assert post.word_count > 50, "Post should be substantial"
```

## 📊 Sample Outputs

### Tech Post (English)
```
Topic: AI in Healthcare: Revolutionizing Medical Diagnosis and Treatment
Language: English
Word Count: 168
Paragraph Count: 3
Has Call-to-Action: True
Technical Depth: Advanced

Generated Post Content:
==================================================
Artificial intelligence (AI) is transforming the healthcare industry by revolutionizing medical diagnosis and treatment. With its ability to analyze vast amounts of data, AI-powered systems can help doctors identify patterns and make predictions that may not be visible to the human eye. This technology is also enabling the development of personalized medicine, where treatments are tailored to individual patients' needs.

AI is also being used to improve patient outcomes by detecting diseases at an early stage, reducing the risk of complications and improving treatment efficacy. Additionally, AI-powered chatbots and virtual assistants are helping to streamline clinical workflows, freeing up healthcare professionals to focus on more complex cases. As a result, patients are receiving more effective care and improved health outcomes.

As we continue to navigate the complexities of modern medicine, it's essential to ask: what role will AI play in your future healthcare journey? Will you be using AI-powered diagnostic tools or virtual assistants to support your care? Let us know in the comments below. #AIinHealthcare #RevolutionizingMedicine #PersonalizedCare
==================================================
```

### General Post (Bengali)
```
Topic: কর্মজীবনে ভারসাম্য রক্ষা করার গুরুত্ব
Language: Bengali
Word Count: 79
Paragraph Count: 3
Has Call-to-Action: True
Content Category: General
Engagement Type: Discussion

Generated Post Content:
==================================================
"কর্মজীবনে ভারসাম্য রক্ষা করার গুরুত্ব

আপনি অবশ্যই জিজ্ঞাসা করেছেন না? তবে, এটি কর্মজীবনের একটি গুরুত্বপূর্ণ দিক। যখন আপনি অতিরিক্ত চালিত হয়ে এবং নিজেকে অতিরিক্ত স্থিতিশীল হিসাবে দেখা শুরু করেন, তখন ভারসাম্য হারানো একটি ঘটনা। আপনি অবশ্যই জিজ্ঞাসা করেছেন তা নির্ভর করে, কোন প্রেক্ষাপটে এই ভারসাম্য বজায় রাখা হচ্ছে। 

আপনি নিজের সীমাবদ্ধতা সম্পর্কে সচেতন হলে, আপনি ভারসাম্য বজায় রাখতে উপযোগী হবেন। কিন্তু এই প্রশ্নটি দিলে, "আপনি কীভাবে সেই অংশগুলিতে জুতা ধরতে পারেন যেখানে ভারসাম্য চালিয়ে যেতে হবে?" #কর্মজীবনে_ভারসাম্য #ভারসাম্য_রক্ষা #সফলতা_এবং_দুর্বলতা"
==================================================
```

## 🆘 Troubleshooting

### Common Issues & Solutions

**❌ Issue: "Ollama not found"**
```bash
# Solution: Install Ollama
# Windows: Download from https://ollama.ai/download
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.ai/install.sh | sh

# Verify installation
ollama --version
```

**❌ Issue: "Model not found"**
```bash
# Solution: Pull the required model
ollama pull llama3.2:3b

# List available models
ollama list
```

**❌ Issue: "Permission denied"**
```bash
# Solution: Fix permissions
# Windows (run as administrator)
# macOS/Linux:
chmod +x examples/*.py
chmod +x src/*.py
```

**❌ Issue: Slow performance**
```bash
# Solutions:
# 1. Check system resources
htop  # or Task Manager on Windows

# 2. Use smaller model
ollama pull llama3.2:1b

# 3. Close other applications
# 4. Ensure Ollama has enough RAM (8GB+ recommended)
```

**❌ Issue: Unicode/Encoding problems**
```python
# Solution: Ensure UTF-8 encoding
with open(filename, 'w', encoding='utf-8') as f:
    f.write(content)
```

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

generator = LinkedInPostGenerator(
    enable_statistics=True,
    log_level="DEBUG"
)
```

### Performance Optimization

```python
# Optimize for speed
generator = LinkedInPostGenerator(
    model_name="llama3.2:1b",           # Smaller model
    classification_temperature=0.0,     # Deterministic
    writing_temperature=0.5,            # Less creative but faster
    enable_statistics=False             # Disable tracking
)
```

## 🚀 Future Roadmap

### Upcoming Features
- [ ] 🎨 **Image Generation** - AI-generated images for posts
- [ ] 📱 **Multi-Platform Support** - Twitter, Facebook, Instagram
- [ ] 🔄 **LinkedIn API Integration** - Direct posting capability
- [ ] 📊 **Analytics Dashboard** - Web-based performance monitoring
- [ ] 🎯 **Custom Model Training** - Fine-tune models for specific industries
- [ ] 🌐 **Additional Languages** - More regional language support
- [ ] 📝 **Template System** - Custom post templates
- [ ] 🔗 **Content Scheduling** - Automated post scheduling

### Contributing to Development

We welcome contributions! See our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 🤝 Contributing

### Development Setup

```bash
# Clone your fork
git clone https://github.com/Rezaul33/AI-Powered-LinkedIn-Post-Generator.git
cd AI-Powered-LinkedIn-Post-Generator

# Create feature branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black src/ examples/
```

### Contribution Guidelines

1. **Code Style**: Follow PEP 8 and use Black for formatting
2. **Testing**: Add tests for new functionality
3. **Documentation**: Update README and docstrings
4. **Commits**: Use clear, descriptive commit messages
5. **PRs**: Include detailed descriptions and test results

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama Team** - For the amazing local AI platform
- **LangChain** - For the powerful LLM framework
- **OpenAI** - For pioneering LLM research
- **LinkedIn Community** - For content best practices

## 📞 Support & Community

### Getting Help
- 📖 **Documentation**: Check this README and demo scripts
- 🐛 **Bug Reports**: [Create an Issue](https://github.com/Rezaul33/AI-Powered-LinkedIn-Post-Generator/issues)
- 💡 **Feature Requests**: [Start a Discussion](https://github.com/Rezaul33/AI-Powered-LinkedIn-Post-Generator/discussions)
- 📧 **Email**: rezaul.islam.da@gmail.com

### Community
- 📱 **LinkedIn**: [Follow me](https://www.linkedin.com/in/md-rezaul-islam-cse/)

---

## 🎉 Ready to Generate Amazing LinkedIn Content!

**⭐ Star this repository** if you find it useful!  
**🍴 Fork and customize** for your specific needs!  
**🔄 Share** with your network and help others discover AI-powered content creation!

**🚀 Start generating professional LinkedIn posts in minutes - completely free and private!**

---

*Built using Python, LangChain, and Ollama*
