import re
import traceback
from typing import List, Dict, Tuple
from .models import AnalysisResult
from functools import lru_cache
import hashlib
import os
import json

_fake_news_pipeline = None
_zero_shot_pipeline = None
HF_AVAILABLE = False
LAST_ERROR = None

# Performance caches with size limits
_content_type_cache = {}
_intent_cache = {}
_response_cache = {}

# Cache management
MAX_CACHE_SIZE = 500

# OpenAI Configuration
try:
    from django.conf import settings
    OPENAI_API_KEY = getattr(settings, 'OPENAI_API_KEY', '') or os.getenv('OPENAI_API_KEY', '')
except:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

OPENAI_AVAILABLE = False
openai_client = None

def initialize_openai():
    """Initialize OpenAI client if API key is available."""
    global OPENAI_AVAILABLE, openai_client
    try:
        if OPENAI_API_KEY:
            from openai import OpenAI
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            OPENAI_AVAILABLE = True
            print("✅ OpenAI API initialized successfully")
        else:
            print("⚠️ OpenAI API key not found. Using fallback responses.")
    except ImportError:
        print("⚠️ OpenAI package not installed. Install with: pip install openai")
    except Exception as e:
        print(f"⚠️ OpenAI initialization failed: {e}")

def get_gpt_response(user_message: str, conversation_history: List[Dict] = None) -> str:
    """
    Get a natural response from GPT for conversational queries.
    """
    global OPENAI_AVAILABLE
    
    if not OPENAI_AVAILABLE:
        initialize_openai()
    
    if not OPENAI_AVAILABLE:
        return generate_fallback_response(user_message)
    
    try:
        global openai_client
        
        # Build conversation context
        messages = [
            {
                "role": "system", 
                "content": """You are a helpful AI assistant integrated into a Fake News Detection system. 

Your primary role is to have natural conversations with users while being aware that you're part of a fact-checking platform. 

Guidelines:
- Answer questions naturally and helpfully like ChatGPT
- Be conversational, friendly, and engaging
- If users ask about fact-checking, mention that you can analyze content they share
- For general questions, provide informative and helpful responses
- Keep responses concise but informative (2-3 paragraphs max for complex topics)
- If asked about your capabilities, mention both conversation and fact-checking features
- Be honest about your limitations
- Maintain a helpful and professional tone"""
            }
        ]
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history[-10:])  # Last 10 messages for context
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Make API call with timeout
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=300,  # Reduced for faster responses
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            timeout=10  # 10 second timeout
        )
        
        gpt_response = response.choices[0].message.content.strip()
        
        # Add a subtle indicator that this is from GPT
        return f"💬 **AI Assistant** - {gpt_response}"
        
    except Exception as e:
        print(f"GPT API Error: {e}")
        return generate_fallback_response(user_message)

def generate_fallback_response(user_message: str) -> str:
    """
    Generate a fallback response when OpenAI is not available.
    """
    text_lower = user_message.lower().strip()
    
    # Smart fallback responses
    if any(phrase in text_lower for phrase in ['how are you', 'what\'s up', 'sup']):
        return "💬 **AI Assistant** - I'm doing well, thank you! I'm here to help with conversations and fact-checking. What's on your mind today?"
    
    elif any(phrase in text_lower for phrase in ['what is', 'what are', 'tell me about', 'explain']):
        return "💬 **AI Assistant** - That's an interesting question! While I'd love to give you a detailed answer, I'm currently running in basic mode. For the best conversational experience, an OpenAI API key would enable more natural responses. However, I'm still great at fact-checking content if you'd like to share something to verify!"
    
    elif any(phrase in text_lower for phrase in ['how to', 'can you help', 'help me']):
        return "💬 **AI Assistant** - I'd be happy to help! While my conversational abilities are limited without an API connection, I excel at fact-checking and information verification. Feel free to share any content you'd like me to analyze, or ask about misinformation and media literacy!"
    
    elif text_lower.endswith('?'):
        return "💬 **AI Assistant** - That's a great question! I'm currently running in basic conversation mode. For more detailed discussions, connecting an OpenAI API key would unlock my full conversational abilities. In the meantime, I'm excellent at fact-checking - feel free to share any content you'd like verified!"
    
    else:
        return "💬 **AI Assistant** - I hear you! While I'm great at having conversations, my responses are more natural with an API connection. However, I'm always ready to help with fact-checking and information verification. What can I help you with today?"

def clear_caches():
    """Clear all caches to free memory."""
    global _content_type_cache, _intent_cache, _response_cache
    _content_type_cache.clear()
    _intent_cache.clear()
    _response_cache.clear()

def manage_cache_size():
    """Keep caches under size limit."""
    if len(_content_type_cache) > MAX_CACHE_SIZE:
        # Remove oldest entries (simple FIFO)
        keys_to_remove = list(_content_type_cache.keys())[:MAX_CACHE_SIZE//2]
        for key in keys_to_remove:
            _content_type_cache.pop(key, None)
    
    if len(_intent_cache) > MAX_CACHE_SIZE:
        keys_to_remove = list(_intent_cache.keys())[:MAX_CACHE_SIZE//2]
        for key in keys_to_remove:
            _intent_cache.pop(key, None)
    
    if len(_response_cache) > MAX_CACHE_SIZE:
        keys_to_remove = list(_response_cache.keys())[:MAX_CACHE_SIZE//2]
        for key in keys_to_remove:
            _response_cache.pop(key, None)

# Compiled regex patterns for better performance
COMPILED_PATTERNS = {
    'greeting': re.compile(r'^(hi|hello|hey|good morning|good afternoon|good evening)', re.IGNORECASE),
    'question': re.compile(r'^(what|who|when|where|why|how|which|can|could|would|will|do|does|did|is|are|was|were)', re.IGNORECASE),
    'news': re.compile(r'(breaking|news|report|article|headline|story|according to|sources say)', re.IGNORECASE),
    'social': re.compile(r'(just saw|can\'t believe|omg|wtf|lol|check this out|viral|trending)', re.IGNORECASE),
    'verification': re.compile(r'(is this true|fact check|verify|real or fake|true\?|fake\?)', re.IGNORECASE),
    'urls': re.compile(r'https?://\S+'),
    'dates': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b'),
    'caps': re.compile(r'\b[A-Z]{3,}\b'),
}


def _initialize_pipelines():
    global _fake_news_pipeline, _zero_shot_pipeline, HF_AVAILABLE, LAST_ERROR
    if _fake_news_pipeline is not None and _zero_shot_pipeline is not None:
        return
    
    # Skip heavy model loading for faster responses - use lightweight heuristics instead
    print("⚡ Using lightweight heuristic models for faster performance")
    HF_AVAILABLE = False
    LAST_ERROR = "Disabled for performance - using fast heuristics"
    
    # Uncomment below to enable heavy models (will be slower)
    # try:
    #     from transformers import pipeline
    #     _fake_news_pipeline = pipeline(
    #         "text-classification",
    #         model="distilbert-base-uncased-finetuned-sst-2-english"
    #     )
    #     _zero_shot_pipeline = pipeline(
    #         "zero-shot-classification",
    #         model="facebook/bart-large-mnli"
    #     )
    #     HF_AVAILABLE = True
    #     LAST_ERROR = None
    # except Exception as e:
    #     HF_AVAILABLE = False
    #     LAST_ERROR = str(e)


def classify_text_type(text: str) -> Tuple[str, float]:
    _initialize_pipelines()
    if not (HF_AVAILABLE and _zero_shot_pipeline):
        return 'unknown', 0.0
    candidate_labels = [
        'news article', 'opinion piece', 'scientific paper', 'social media post',
        'personal message', 'legal document', 'product description', 'job posting',
        'poem', 'song lyrics', 'simple question or greeting'
    ]
    try:
        result = _zero_shot_pipeline(text, candidate_labels, multi_label=False)
        return result['labels'][0], float(result['scores'][0])
    except Exception:
        return 'unknown', 0.0


def _heuristic_detector(text: str) -> Tuple[str, float]:
    text_lower = text.lower()
    suspicious = [
        'allegedly', 'rumor', 'rumour', 'unverified', 'fake', 'hoax',
        'conspiracy', 'shocking', 'claims', 'reported', 'sources say',
        'according to sources', 'exclusive', 'must read'
    ]
    score = 0.5
    for word in suspicious:
        if word in text_lower:
            score -= 0.06
    score -= min(0.15, 0.02 * text.count('!'))
    score -= min(0.12, 0.04 * len(re.findall(r'\b[A-Z]{2,}\b', text)))
    word_count = len(text.split())
    if word_count < 5:
        score -= 0.08
    if word_count > 1000:
        score -= 0.05
    score = max(0.01, min(0.99, score))
    verdict = 'Real' if score >= 0.5 else 'Fake'
    confidence = max(0.01, min(0.99, abs(score - 0.5) * 2))
    return verdict, confidence


def detect_fake_news(text: str) -> Tuple[str, float]:
    _initialize_pipelines()
    if HF_AVAILABLE and _fake_news_pipeline is not None:
        try:
            result = _fake_news_pipeline(text)[0]
            label = str(result.get('label', '')).upper()
            score = float(result.get('score', 0.0))
            return ('Real' if label in ['POSITIVE', 'REAL', 'LABEL_1'] else 'Fake', score)
        except Exception:
            return _heuristic_detector(text)
    return _heuristic_detector(text)


def search_fact_check_sources(query: str) -> List[Dict[str, str]]:
    """
    Fast fact-check source search with reduced timeout for better performance.
    Returns a list of {site, title, url, snippet}.
    """
    import requests
    from bs4 import BeautifulSoup
    import concurrent.futures
    import threading

    results: List[Dict[str, str]] = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    def search_snopes():
        try:
            response = requests.get(
                "https://www.snopes.com/search/", 
                params={"q": query[:120]}, 
                headers=headers, 
                timeout=3  # Reduced timeout
            )
            if response.ok:
                soup = BeautifulSoup(response.text, 'html.parser')
                for item in soup.select('article a.media-wrapper')[:2]:  # Reduced results
                    url = item.get('href')
                    title = (item.get('title') or item.text or '').strip()
                    if url and title:
                        results.append({
                            'site': 'Snopes', 'title': title, 'url': url, 'snippet': ''
                        })
        except Exception:
            pass
    
    def search_politifact():
        try:
            response = requests.get(
                "https://www.politifact.com/search/", 
                params={"q": query[:120]}, 
                headers=headers, 
                timeout=3  # Reduced timeout
            )
            if response.ok:
                soup = BeautifulSoup(response.text, 'html.parser')
                for item in soup.select('article a')[:2]:  # Reduced results
                    href = item.get('href')
                    if href and href.startswith('/'):
                        url = f"https://www.politifact.com{href}"
                        title = (item.get('title') or item.text or '').strip()
                        if title:
                            results.append({
                                'site': 'PolitiFact', 'title': title, 'url': url, 'snippet': ''
                            })
        except Exception:
            pass
    
    # Run searches in parallel with timeout
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(search_snopes),
            executor.submit(search_politifact)
        ]
        
        # Wait for completion with overall timeout
        try:
            concurrent.futures.wait(futures, timeout=5)  # 5 second max total
        except Exception:
            pass
    
    # Add a static Wikipedia link as fallback (no scraping)
    if query and len(query.strip()) > 3:
        wiki_query = query.replace(' ', '_')[:50]
        results.append({
            'site': 'Wikipedia', 
            'title': f'Search Wikipedia for: {query[:30]}...', 
            'url': f'https://en.wikipedia.org/wiki/Special:Search/{wiki_query}', 
            'snippet': ''
        })
    
    # Deduplicate by URL
    seen = set()
    unique: List[Dict[str, str]] = []
    for r in results:
        if r['url'] not in seen:
            unique.append(r)
            seen.add(r['url'])
    return unique[:3]  # Reduced to 3 results max


@lru_cache(maxsize=1000)
def analyze_content_structure_fast(text: str) -> Dict[str, object]:
    """
    Fast optimized content structure analysis with caching.
    """
    text_clean = text.strip()
    words = text_clean.split()
    
    analysis = {
        'word_count': len(words),
        'has_urls': bool(COMPILED_PATTERNS['urls'].search(text_clean)),
        'has_quotes': '"' in text_clean or "'" in text_clean,
        'has_dates': bool(COMPILED_PATTERNS['dates'].search(text_clean)),
        'has_caps': bool(COMPILED_PATTERNS['caps'].search(text_clean)),
        'exclamation_count': text_clean.count('!'),
        'question_count': text_clean.count('?'),
        'is_question': text_clean.endswith('?'),
    }
    
    return analysis


def get_text_hash(text: str) -> str:
    """Generate a hash for caching purposes."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def classify_content_type_fast(text: str) -> Tuple[str, float, Dict[str, object]]:
    """
    Fast optimized content classification with caching.
    """
    # Check cache first
    text_hash = get_text_hash(text)
    if text_hash in _content_type_cache:
        return _content_type_cache[text_hash]
    
    text_clean = text.strip()
    text_lower = text_clean.lower()
    
    # Get fast structural analysis
    structure = analyze_content_structure_fast(text_clean)
    
    # Fast pattern matching using compiled regex
    scores = {
        'greeting': 0,
        'question': 0,
        'news_article': 0,
        'social_media_post': 0,
        'claim_verification': 0,
        'conversation': 0
    }
    
    # Quick pattern checks
    if COMPILED_PATTERNS['greeting'].match(text_lower):
        scores['greeting'] = 3.0
    
    if COMPILED_PATTERNS['question'].match(text_lower) or structure['is_question']:
        scores['question'] = 2.0
    
    if COMPILED_PATTERNS['news'].search(text_lower):
        scores['news_article'] = 2.5
    
    if COMPILED_PATTERNS['social'].search(text_lower):
        scores['social_media_post'] = 2.0
    
    if COMPILED_PATTERNS['verification'].search(text_lower):
        scores['claim_verification'] = 3.0
    
    # Apply quick structural bonuses
    if structure['word_count'] > 100:
        scores['news_article'] += 1.0
    
    if structure['has_urls']:
        scores['news_article'] += 1.5
        scores['claim_verification'] += 1.0
    
    if structure['exclamation_count'] > 2:
        scores['social_media_post'] += 1.0
    
    # Find best type
    if max(scores.values()) == 0:
        result = ('conversation', 0.3, structure)
    else:
        best_type = max(scores, key=scores.get)
        max_score = scores[best_type]
        confidence = min(0.95, max_score / 5.0 + 0.3)  # Simplified confidence
        result = (best_type, confidence, structure)
    
    # Cache result with size management
    manage_cache_size()
    _content_type_cache[text_hash] = result
    return result


def classify_user_intent_fast(text: str) -> Tuple[str, float]:
    """
    Enhanced intent classification following the 5-category system:
    1. Text → News statements, paragraphs to check
    2. Image → Picture uploads (handled by frontend)
    3. Video → Video uploads/links (handled by frontend)
    4. URL → Links to verify
    5. General Chat → Greetings, questions, casual conversation
    """
    # Check cache first
    text_hash = get_text_hash(text)
    if text_hash in _intent_cache:
        return _intent_cache[text_hash]
    
    text_lower = text.lower().strip()
    text_clean = text.strip()
    
    # Category 4: URL Detection (Highest Priority)
    if COMPILED_PATTERNS['urls'].search(text_clean):
        result = ('analysis', 0.95)
        print(f"🔗 URL detected: This looks like a URL, I will check it.")
    
    # Category 5: General Chat Detection (More Inclusive)
    elif len(text_lower) < 3:
        result = ('conversation', 0.9)
        print(f"💬 Short input: This seems to be a general question, let's chat.")
    
    elif COMPILED_PATTERNS['greeting'].match(text_lower):
        result = ('conversation', 0.95)
        print(f"👋 Greeting detected: This seems to be a general question, let's chat.")
    
    elif any(phrase in text_lower for phrase in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'how are you', 'what\'s up', 'sup', 'yo', 'hola']):
        result = ('conversation', 0.9)
        print(f"💬 Casual greeting: This seems to be a general question, let's chat.")
    
    elif any(phrase in text_lower for phrase in ['help', 'what can you do', 'how do you work', 'thank you', 'thanks', 'bye', 'goodbye', 'see you', 'nice', 'cool', 'awesome', 'great', 'ok', 'okay']):
        result = ('conversation', 0.85)
        print(f"💬 General question: This seems to be a general question, let's chat.")
    
    # Recognize more conversational patterns
    elif any(phrase in text_lower for phrase in ['tell me about', 'what do you think', 'i think', 'i believe', 'my opinion', 'in my view', 'personally']):
        result = ('conversation', 0.8)
        print(f"💬 Opinion/Discussion: This seems to be a general question, let's chat.")
    
    elif any(phrase in text_lower for phrase in ['how to', 'can you explain', 'what is', 'what are', 'who is', 'where is', 'when is', 'why is']):
        result = ('conversation', 0.75)
        print(f"💬 General inquiry: This seems to be a general question, let's chat.")
    
    elif text_lower.endswith('?') and len(text.split()) < 20 and not any(word in text_lower for word in ['true', 'fake', 'real', 'verify', 'check', 'analyze', 'fact']):
        result = ('conversation', 0.8)
        print(f"❓ Question: This seems to be a general question, let's chat.")
    
    # Catch casual statements and responses
    elif len(text.split()) < 10 and not any(word in text_lower for word in ['breaking', 'news', 'report', 'according', 'sources', 'claims', 'study']):
        result = ('conversation', 0.7)
        print(f"💬 Casual statement: This seems to be a general question, let's chat.")
    
    # Category 1: Text Analysis Detection
    elif COMPILED_PATTERNS['verification'].search(text_lower):
        result = ('analysis', 0.95)
        print(f"🔍 Verification request: This looks like text content, I will analyze it.")
    
    elif any(phrase in text_lower for phrase in ['is this true', 'fact check', 'verify this', 'real or fake', 'check this', 'analyze this']):
        result = ('analysis', 0.9)
        print(f"📊 Analysis request: This looks like text content, I will analyze it.")
    
    elif COMPILED_PATTERNS['news'].search(text_lower):
        result = ('analysis', 0.85)
        print(f"📰 News content: This looks like text content, I will analyze it.")
    
    elif len(text.split()) > 30:  # Long text likely needs analysis
        result = ('analysis', 0.8)
        print(f"📄 Long text: This looks like text content, I will analyze it.")
    
    elif any(word in text_lower for word in ['breaking', 'report', 'according to', 'sources say', 'claims', 'study shows', 'research']):
        result = ('analysis', 0.75)
        print(f"📋 News/Research content: This looks like text content, I will analyze it.")
    
    # Default to conversation for unclear inputs
    else:
        result = ('conversation', 0.6)
        print(f"💭 Unclear input: This seems to be a general question, let's chat.")
    
    # Cache and return with size management
    manage_cache_size()
    _intent_cache[text_hash] = result
    return result


# Keep the original function as backup
def classify_user_intent(text: str) -> Tuple[str, float]:
    """Fallback to fast version for better performance."""
    return classify_user_intent_fast(text)


def generate_conversational_response_fast(text: str, conversation_history: List[Dict] = None) -> str:
    """
    Enhanced LLM-like conversational response system with intelligent context understanding.
    """
    text_clean = text.strip()
    text_lower = text_clean.lower()
    
    # Category 4: URL Detection - Always handle this locally
    if COMPILED_PATTERNS['urls'].search(text_clean):
        return "🔗 **URL Detected** - This looks like a URL, I will check it for you! Let me analyze the source credibility and content accuracy."
    
    # For conversational queries, try GPT first if available
    if OPENAI_AVAILABLE or OPENAI_API_KEY:
        try:
            gpt_response = get_gpt_response(text_clean, conversation_history)
            return gpt_response
        except Exception as e:
            print(f"GPT fallback triggered: {e}")
            # Continue to intelligent fallback responses below
    
    # Enhanced LLM-like fallback responses with intelligent context understanding
    response = generate_intelligent_response(text_clean, text_lower)
    return response


def generate_intelligent_response(text_clean: str, text_lower: str) -> str:
    """
    Generate intelligent, context-aware responses like an LLM.
    This function analyzes the user's input and provides appropriate responses.
    """
    import random
    
    # Analyze the input for better context understanding
    words = text_clean.split()
    word_count = len(words)
    
    # 1. GREETINGS - Natural and welcoming
    if COMPILED_PATTERNS['greeting'].match(text_lower) or any(phrase in text_lower for phrase in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
        responses = [
            "👋 Hello! I'm your AI assistant specializing in fact-checking and information verification. I'm here to help you navigate the world of information. What's on your mind today?",
            "Hi there! Great to meet you! I can help with fact-checking, analyzing content, or just having a conversation about information and media literacy. How can I assist you?",
            "Hey! I'm your friendly AI fact-checker. Whether you want to verify some information, discuss current events, or just chat, I'm here for you. What would you like to talk about?"
        ]
        return random.choice(responses)
    
    # 2. READINESS/CAPABILITY QUESTIONS - Show understanding
    if any(phrase in text_lower for phrase in ['are you ready', 'ready to', 'can you', 'are you able', 'do you understand']):
        if 'converse' in text_lower or 'talk' in text_lower or 'chat' in text_lower:
            return "💬 Absolutely! I'm ready to have a conversation with you. I can discuss various topics, help with fact-checking, analyze information, or just chat about whatever interests you. What would you like to talk about?"
        elif 'analyze' in text_lower or 'check' in text_lower or 'verify' in text_lower:
            return "🔍 Yes, I'm definitely ready to analyze content for you! I can fact-check news articles, verify claims, analyze social media posts, check website credibility, and much more. Just share what you'd like me to examine."
        else:
            return "✅ I'm ready and eager to help! I can engage in conversations, fact-check information, analyze content, discuss current events, or assist with media literacy questions. What specific task do you have in mind?"
    
    # 3. ANALYSIS REQUESTS - Show enthusiasm and capability
    if any(phrase in text_lower for phrase in ['analyze', 'check this', 'verify', 'fact check', 'is this true', 'examine']):
        if 'sentence' in text_lower or 'statement' in text_lower:
            return "📝 I'd be happy to analyze a sentence or statement for you! Please go ahead and share the text you'd like me to examine. I'll check its credibility, look for potential misinformation patterns, and provide you with a detailed analysis."
        else:
            return "🔍 Excellent! I'm great at analyzing content. Please share the text, article, claim, or any information you'd like me to fact-check. I'll examine it thoroughly and provide you with insights about its reliability and accuracy."
    
    # 4. QUESTIONS ABOUT CAPABILITIES - Detailed and helpful
    if any(phrase in text_lower for phrase in ['what can you do', 'how do you work', 'what are your capabilities', 'help me understand']):
        return """🤖 I'm an advanced AI assistant with several key capabilities:

**💬 Intelligent Conversation**: I can engage in natural discussions about various topics, understand context, and provide thoughtful responses.

**🔍 Fact-Checking**: I analyze text for misinformation patterns, verify claims against reliable sources, and assess credibility.

**📊 Content Analysis**: I can examine news articles, social media posts, statements, and other content to determine reliability.

**🌐 Source Verification**: I check website credibility, cross-reference information, and identify potential bias.

**🧠 Context Understanding**: Unlike simple chatbots, I understand nuance, context, and can distinguish between different types of content.

I'm designed to be helpful, accurate, and conversational. What would you like to explore together?"""
    
    # 5. AFFIRMATIVE RESPONSES - Show understanding and engagement
    if any(phrase in text_lower for phrase in ['yes', 'yeah', 'yep', 'correct', 'right', 'exactly', 'that\'s right']):
        return "👍 Great! I'm glad we're on the same page. What would you like to do next? I'm here to help with fact-checking, analysis, or just continue our conversation."
    
    # 6. QUESTIONS ENDING WITH ? - Contextual responses
    if text_clean.endswith('?'):
        if word_count <= 3:  # Short questions
            return "🤔 That's an interesting question! Could you provide a bit more context or detail? I'd love to give you a thoughtful and helpful response."
        elif any(word in text_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            return "💭 That's a great question! I'd be happy to help you explore that topic. Could you share more details about what specifically you'd like to know? The more context you provide, the better I can assist you."
        else:
            return "❓ Interesting question! I'm here to help you find answers. Whether it's about fact-checking, information verification, or general discussion, I'm ready to dive into the topic with you."
    
    # 7. STATEMENTS ABOUT CONVERSATION/COMMUNICATION
    if any(phrase in text_lower for phrase in ['talk', 'discuss', 'conversation', 'communicate', 'speak']):
        return "💬 I'd love to have a conversation with you! I enjoy discussing various topics - from current events and media literacy to fact-checking and information analysis. What's something you're curious about or would like to explore together?"
    
    # 8. EXPRESSIONS OF UNDERSTANDING OR AGREEMENT
    if any(phrase in text_lower for phrase in ['i see', 'i understand', 'got it', 'makes sense', 'i think', 'i believe']):
        return "💡 I appreciate you sharing your thoughts! It's great when we can have meaningful exchanges. Is there anything specific you'd like to discuss further or any information you'd like me to help verify?"
    
    # 9. REQUESTS FOR HELP OR ASSISTANCE
    if any(phrase in text_lower for phrase in ['help', 'assist', 'support', 'guide']):
        return "🤝 I'm absolutely here to help! Whether you need fact-checking, want to verify information, discuss current events, or just have a conversation, I'm ready to assist. What specific area would you like help with?"
    
    # 10. EXPRESSIONS OF FRUSTRATION OR CONFUSION
    if any(phrase in text_lower for phrase in ['not working', 'doesn\'t understand', 'not responding', 'confused', 'frustrated']):
        return "😊 I understand your concern, and I'm here to help! I'm designed to be conversational and responsive to what you're saying. Let's start fresh - what would you like to talk about or what can I help you with today?"
    
    # 11. LONGER STATEMENTS - Show engagement
    if word_count > 10:
        return "🧠 I can see you've shared some detailed thoughts with me. I'm processing what you've said and I'm ready to engage with the topic. Could you let me know what specific aspect you'd like me to focus on or how I can best help you with this?"
    
    # 12. SHORT RESPONSES - Encourage elaboration
    if word_count <= 2:
        responses = [
            "💭 I'm listening! Feel free to elaborate on what you're thinking about.",
            "🤔 Interesting! Could you tell me more about what's on your mind?",
            "💬 I'm here and ready to chat! What would you like to discuss?",
            "✨ I'm engaged and ready to help! What can we explore together?"
        ]
        return random.choice(responses)
    
    # 13. DEFAULT INTELLIGENT RESPONSE - Context-aware and engaging
    responses = [
        "💭 That's interesting! I'm here to engage with whatever you'd like to discuss. Whether it's fact-checking, analyzing information, or just having a conversation, I'm ready to help. What's on your mind?",
        "🤖 I'm processing what you've shared and I'm ready to respond thoughtfully. Could you let me know what specific aspect you'd like me to focus on or how I can best assist you?",
        "💬 I appreciate you sharing that with me! I'm designed to be conversational and helpful. What would you like to explore together - fact-checking, discussion, analysis, or something else?",
        "🧠 I'm engaged and ready to dive deeper into this topic with you. What specific questions do you have or what would you like me to help you with?",
        "✨ Thanks for that input! I'm here to provide thoughtful, contextual responses. What direction would you like our conversation to take?"
    ]
    return random.choice(responses)


# Update the main function to use fast version with GPT
def generate_conversational_response(text: str, conversation_history: List[Dict] = None) -> str:
    """Use enhanced version with GPT integration."""
    return generate_conversational_response_fast(text, conversation_history)

# Expose fast functions for direct import
def classify_content_type(text: str) -> Tuple[str, float, Dict[str, object]]:
    """Use fast version for better performance."""
    return classify_content_type_fast(text)

def analyze_content_structure(text: str) -> Dict[str, object]:
    """Use fast version for better performance."""
    return analyze_content_structure_fast(text)


def generate_community_notes(text: str, verdict: str) -> Dict[str, object]:
    """
    Returns a dict with 'summary' and 'sources' fields.
    """
    base_summary = (
        "This content appears credible based on an automated check."
        if verdict == 'Real' else
        "This content may be misleading based on automated signals. Consider the sources below."
    )
    sources = search_fact_check_sources(text[:300])
    return {"summary": base_summary, "sources": sources}

# Initialize OpenAI when module is loaded
try:
    initialize_openai()
except Exception as e:
    print(f"OpenAI initialization deferred: {e}")
