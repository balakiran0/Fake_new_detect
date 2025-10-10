"""
Enhanced Fake News Verification Engine
Implements comprehensive 8-step verification pipeline with trusted source integration
"""

import re
import json
import time
import requests
from typing import List, Dict, Tuple, Optional
from bs4 import BeautifulSoup
import concurrent.futures
from functools import lru_cache
import hashlib
from datetime import datetime
import os

# OpenAI Integration
try:
    from django.conf import settings
    OPENAI_API_KEY = getattr(settings, 'OPENAI_API_KEY', '') or os.getenv('OPENAI_API_KEY', '')
except:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

OPENAI_AVAILABLE = False
openai_client = None

def initialize_openai():
    """Initialize OpenAI client for enhanced verification."""
    global OPENAI_AVAILABLE, openai_client
    try:
        if OPENAI_API_KEY:
            from openai import OpenAI
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            OPENAI_AVAILABLE = True
            print("✅ Enhanced Verification: OpenAI initialized")
        else:
            print("⚠️ Enhanced Verification: Using heuristic fallbacks")
    except Exception as e:
        print(f"⚠️ Enhanced Verification: {e}")

# Initialize on import
initialize_openai()

# Trusted News Sources Configuration
TRUSTED_SOURCES = {
    'fact_checkers': [
        {'name': 'Snopes', 'url': 'https://www.snopes.com/search/', 'weight': 0.9},
        {'name': 'PolitiFact', 'url': 'https://www.politifact.com/search/', 'weight': 0.9},
        {'name': 'FactCheck.org', 'url': 'https://www.factcheck.org/', 'weight': 0.9},
        {'name': 'AltNews', 'url': 'https://www.altnews.in/', 'weight': 0.8},
        {'name': 'BOOM Live', 'url': 'https://www.boomlive.in/', 'weight': 0.8},
    ],
    'news_outlets': [
        {'name': 'Reuters', 'url': 'https://www.reuters.com/search/news?blob=', 'weight': 0.95},
        {'name': 'AP News', 'url': 'https://apnews.com/search?q=', 'weight': 0.95},
        {'name': 'BBC', 'url': 'https://www.bbc.com/search?q=', 'weight': 0.9},
        {'name': 'The Hindu', 'url': 'https://www.thehindu.com/search/?q=', 'weight': 0.85},
        {'name': 'NYTimes', 'url': 'https://www.nytimes.com/search?query=', 'weight': 0.9},
    ]
}

# Performance cache
verification_cache = {}
MAX_CACHE_SIZE = 200

class EnhancedVerificationEngine:
    """
    Comprehensive fake news verification following 8-step pipeline
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def verify_content(self, text: str, use_openai: bool = True) -> Dict:
        """
        Main verification pipeline - implements all 8 steps
        """
        start_time = time.time()
        
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        if text_hash in verification_cache:
            cached_result = verification_cache[text_hash].copy()
            cached_result['cached'] = True
            cached_result['processing_time'] = f"<0.1s (cached)"
            return cached_result
        
        try:
            # Step 1: Extract Core Claim
            core_claim = self.extract_core_claim(text, use_openai)
            
            # Step 2: Extract Key Entities
            entities = self.extract_entities(core_claim, use_openai)
            
            # Step 3: Search Trusted Sources
            search_results = self.search_trusted_sources(core_claim)
            
            # Step 4: Gather Evidence
            evidence = self.gather_evidence(search_results, core_claim)
            
            # Step 5: Cross-Verification
            verification_status = self.cross_verify_claim(core_claim, evidence)
            
            # Step 6: Calculate Consensus
            verdict, confidence = self.calculate_consensus(verification_status, evidence)
            
            # Step 7: Bias/Context Check
            bias_analysis = self.analyze_bias_context(evidence, search_results)
            
            # Step 8: Generate Final Output
            result = self.generate_final_output(
                original_text=text,
                core_claim=core_claim,
                entities=entities,
                evidence=evidence,
                verdict=verdict,
                confidence=confidence,
                bias_analysis=bias_analysis,
                processing_time=time.time() - start_time
            )
            
            # Cache result
            self.manage_cache()
            verification_cache[text_hash] = result.copy()
            
            return result
            
        except Exception as e:
            return self.generate_error_response(str(e), time.time() - start_time)
    
    def extract_core_claim(self, text: str, use_openai: bool = True) -> str:
        """Step 1: Extract core claim from news/article/claim"""
        if use_openai and OPENAI_AVAILABLE:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{
                        "role": "system",
                        "content": "Extract the core claim from the given text in 1-2 sentences, removing opinions or exaggerations. Focus on factual statements that can be verified."
                    }, {
                        "role": "user",
                        "content": f"Given this news/article/claim: \"{text[:1000]}\", extract the core claim."
                    }],
                    max_tokens=150,
                    temperature=0.1,
                    timeout=8
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"OpenAI claim extraction failed: {e}")
        
        # Fallback heuristic extraction
        return self.extract_claim_heuristic(text)
    
    def extract_claim_heuristic(self, text: str) -> str:
        """Heuristic claim extraction when OpenAI unavailable"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Look for key claim indicators
        claim_indicators = ['claims', 'reports', 'announces', 'says', 'states', 'reveals', 'confirms']
        
        for sentence in sentences[:5]:  # Check first 5 sentences
            if any(indicator in sentence.lower() for indicator in claim_indicators):
                return sentence[:200] + ('...' if len(sentence) > 200 else '')
        
        # Return first substantial sentence if no indicators found
        return sentences[0][:200] + ('...' if len(sentences[0]) > 200 else '') if sentences else text[:200]
    
    def extract_entities(self, claim: str, use_openai: bool = True) -> Dict[str, List[str]]:
        """Step 2: Extract key entities (people, places, dates, organizations, numbers)"""
        if use_openai and OPENAI_AVAILABLE:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{
                        "role": "system",
                        "content": "Extract key entities from the claim and return as JSON with keys: people, places, dates, organizations, numbers. Return only the JSON object."
                    }, {
                        "role": "user",
                        "content": f"From the claim \"{claim}\", identify key entities."
                    }],
                    max_tokens=200,
                    temperature=0.1,
                    timeout=8
                )
                return json.loads(response.choices[0].message.content.strip())
            except Exception as e:
                print(f"OpenAI entity extraction failed: {e}")
        
        # Fallback heuristic extraction
        return self.extract_entities_heuristic(claim)
    
    def extract_entities_heuristic(self, claim: str) -> Dict[str, List[str]]:
        """Heuristic entity extraction"""
        entities = {
            'people': [],
            'places': [],
            'dates': [],
            'organizations': [],
            'numbers': []
        }
        
        # Extract dates
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        ]
        for pattern in date_patterns:
            entities['dates'].extend(re.findall(pattern, claim))
        
        # Extract numbers
        entities['numbers'] = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', claim)
        
        # Extract potential organizations (capitalized words)
        org_pattern = r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b'
        potential_orgs = re.findall(org_pattern, claim)
        entities['organizations'] = [org for org in potential_orgs if len(org.split()) <= 3]
        
        return entities
    
    def search_trusted_sources(self, query: str) -> List[Dict]:
        """Step 3: Search multiple trusted sources"""
        results = []
        search_query = query[:100]  # Limit query length
        
        def search_source(source_info, source_type):
            try:
                if source_type == 'fact_checkers':
                    return self.search_fact_checker(source_info, search_query)
                else:
                    return self.search_news_outlet(source_info, search_query)
            except Exception as e:
                print(f"Search failed for {source_info['name']}: {e}")
                return []
        
        # Parallel search with timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            # Search fact checkers
            for source in TRUSTED_SOURCES['fact_checkers'][:3]:
                futures.append(executor.submit(search_source, source, 'fact_checkers'))
            
            # Search news outlets
            for source in TRUSTED_SOURCES['news_outlets'][:3]:
                futures.append(executor.submit(search_source, source, 'news_outlets'))
            
            # Collect results with timeout
            for future in concurrent.futures.as_completed(futures, timeout=10):
                try:
                    source_results = future.result(timeout=3)
                    results.extend(source_results)
                except Exception as e:
                    continue
        
        return results[:10]  # Limit to top 10 results
    
    def search_fact_checker(self, source_info: Dict, query: str) -> List[Dict]:
        """Search individual fact-checking source"""
        results = []
        try:
            if source_info['name'] == 'Snopes':
                response = self.session.get(
                    "https://www.snopes.com/search/",
                    params={"q": query},
                    timeout=5
                )
                if response.ok:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for item in soup.select('article a.media-wrapper')[:2]:
                        url = item.get('href')
                        title = (item.get('title') or item.text or '').strip()
                        if url and title:
                            results.append({
                                'source': source_info['name'],
                                'title': title,
                                'url': url,
                                'type': 'fact_check',
                                'weight': source_info['weight']
                            })
            
            elif source_info['name'] == 'PolitiFact':
                response = self.session.get(
                    "https://www.politifact.com/search/",
                    params={"q": query},
                    timeout=5
                )
                if response.ok:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for item in soup.select('article a')[:2]:
                        href = item.get('href')
                        if href and href.startswith('/'):
                            url = f"https://www.politifact.com{href}"
                            title = (item.get('title') or item.text or '').strip()
                            if title:
                                results.append({
                                    'source': source_info['name'],
                                    'title': title,
                                    'url': url,
                                    'type': 'fact_check',
                                    'weight': source_info['weight']
                                })
        except Exception as e:
            pass
        
        return results
    
    def search_news_outlet(self, source_info: Dict, query: str) -> List[Dict]:
        """Search news outlet - simplified implementation"""
        # For demo purposes, return structured search links
        return [{
            'source': source_info['name'],
            'title': f"Search {source_info['name']} for: {query[:50]}...",
            'url': f"{source_info['url']}{query.replace(' ', '+')}",
            'type': 'news_search',
            'weight': source_info['weight']
        }]
    
    def gather_evidence(self, search_results: List[Dict], claim: str) -> List[Dict]:
        """Step 4: Gather evidence from search results"""
        evidence = []
        
        for result in search_results[:5]:  # Process top 5 results
            try:
                if result['type'] == 'fact_check':
                    # Try to extract verdict from fact-check pages
                    evidence_item = {
                        'source': result['source'],
                        'title': result['title'],
                        'url': result['url'],
                        'type': result['type'],
                        'weight': result['weight'],
                        'verdict': self.extract_verdict_from_title(result['title']),
                        'relevance': self.calculate_relevance(claim, result['title'])
                    }
                    evidence.append(evidence_item)
                else:
                    # For news searches, just add the search link
                    evidence.append({
                        'source': result['source'],
                        'title': result['title'],
                        'url': result['url'],
                        'type': result['type'],
                        'weight': result['weight'],
                        'verdict': 'search_link',
                        'relevance': 0.5
                    })
            except Exception as e:
                continue
        
        return evidence
    
    def extract_verdict_from_title(self, title: str) -> str:
        """Extract verdict from fact-check article title"""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['false', 'fake', 'misleading', 'incorrect', 'debunked']):
            return 'FAKE'
        elif any(word in title_lower for word in ['true', 'correct', 'accurate', 'confirmed']):
            return 'REAL'
        elif any(word in title_lower for word in ['mixed', 'partly', 'partially']):
            return 'MIXED'
        else:
            return 'UNVERIFIED'
    
    def calculate_relevance(self, claim: str, title: str) -> float:
        """Calculate relevance score between claim and evidence title"""
        claim_words = set(claim.lower().split())
        title_words = set(title.lower().split())
        
        if not claim_words or not title_words:
            return 0.0
        
        intersection = claim_words.intersection(title_words)
        union = claim_words.union(title_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def cross_verify_claim(self, claim: str, evidence: List[Dict]) -> Dict:
        """Step 5: Cross-verification against evidence"""
        verification = {
            'confirmed_by': [],
            'denied_by': [],
            'mixed_by': [],
            'unverified_by': []
        }
        
        for item in evidence:
            verdict = item.get('verdict', 'UNVERIFIED')
            source_info = {
                'source': item['source'],
                'weight': item['weight'],
                'relevance': item['relevance']
            }
            
            if verdict == 'REAL':
                verification['confirmed_by'].append(source_info)
            elif verdict == 'FAKE':
                verification['denied_by'].append(source_info)
            elif verdict == 'MIXED':
                verification['mixed_by'].append(source_info)
            else:
                verification['unverified_by'].append(source_info)
        
        return verification
    
    def calculate_consensus(self, verification: Dict, evidence: List[Dict]) -> Tuple[str, str]:
        """Step 6: Calculate consensus based on verification results"""
        confirmed_count = len(verification['confirmed_by'])
        denied_count = len(verification['denied_by'])
        mixed_count = len(verification['mixed_by'])
        
        # Weight by source reliability and relevance
        confirmed_weight = sum(item['weight'] * item['relevance'] for item in verification['confirmed_by'])
        denied_weight = sum(item['weight'] * item['relevance'] for item in verification['denied_by'])
        
        # Apply consensus rules
        if denied_count >= 3 or denied_weight >= 2.0:
            return "FAKE", "HIGH" if denied_count >= 3 else "MEDIUM"
        elif confirmed_count >= 3 or confirmed_weight >= 2.0:
            return "REAL", "HIGH" if confirmed_count >= 3 else "MEDIUM"
        elif mixed_count >= 2:
            return "MIXED", "MEDIUM"
        elif denied_count > confirmed_count:
            return "FAKE", "LOW"
        elif confirmed_count > denied_count:
            return "REAL", "LOW"
        else:
            return "UNVERIFIED", "LOW"
    
    def analyze_bias_context(self, evidence: List[Dict], search_results: List[Dict]) -> Dict:
        """Step 7: Analyze bias and context"""
        analysis = {
            'source_diversity': len(set(item['source'] for item in evidence)),
            'fact_check_coverage': len([item for item in evidence if item['type'] == 'fact_check']),
            'news_coverage': len([item for item in evidence if item['type'] == 'news_search']),
            'suspicious_patterns': [],
            'reliability_score': 0.0
        }
        
        # Calculate reliability score
        if evidence:
            total_weight = sum(item['weight'] * item['relevance'] for item in evidence)
            analysis['reliability_score'] = min(1.0, total_weight / len(evidence))
        
        # Check for suspicious patterns
        if analysis['source_diversity'] < 2:
            analysis['suspicious_patterns'].append("Limited source diversity")
        
        if analysis['fact_check_coverage'] == 0:
            analysis['suspicious_patterns'].append("No fact-checker coverage found")
        
        return analysis
    
    def generate_final_output(self, **kwargs) -> Dict:
        """Step 8: Generate structured JSON output"""
        return {
            "verdict": kwargs['verdict'],
            "confidence": kwargs['confidence'],
            "evidence": [
                {
                    "source": item['source'],
                    "title": item['title'],
                    "url": item['url'],
                    "type": item['type'],
                    "verdict": item.get('verdict', 'UNVERIFIED'),
                    "weight": item['weight']
                }
                for item in kwargs['evidence']
            ],
            "explanation": self.generate_explanation(
                kwargs['verdict'], 
                kwargs['confidence'], 
                kwargs['evidence']
            ),
            "core_claim": kwargs['core_claim'],
            "entities": kwargs['entities'],
            "bias_analysis": kwargs['bias_analysis'],
            "processing_time": f"{kwargs['processing_time']:.2f}s",
            "timestamp": datetime.now().isoformat(),
            "cached": False
        }
    
    def generate_explanation(self, verdict: str, confidence: str, evidence: List[Dict]) -> str:
        """Generate human-readable explanation"""
        if verdict == "FAKE":
            return f"This claim appears to be false based on analysis of {len(evidence)} sources. Multiple fact-checkers have identified issues with this information."
        elif verdict == "REAL":
            return f"This claim appears to be accurate based on verification against {len(evidence)} trusted sources."
        elif verdict == "MIXED":
            return f"This claim contains both accurate and inaccurate elements. Further verification recommended."
        else:
            return f"Insufficient reliable evidence found to verify this claim. Consider seeking additional sources."
    
    def generate_error_response(self, error: str, processing_time: float) -> Dict:
        """Generate error response in consistent format"""
        return {
            "verdict": "ERROR",
            "confidence": "LOW",
            "evidence": [],
            "explanation": f"Verification failed due to: {error}",
            "core_claim": "",
            "entities": {},
            "bias_analysis": {},
            "processing_time": f"{processing_time:.2f}s",
            "timestamp": datetime.now().isoformat(),
            "cached": False,
            "error": error
        }
    
    def manage_cache(self):
        """Manage cache size to prevent memory issues"""
        if len(verification_cache) > MAX_CACHE_SIZE:
            # Remove oldest entries
            keys_to_remove = list(verification_cache.keys())[:MAX_CACHE_SIZE//2]
            for key in keys_to_remove:
                verification_cache.pop(key, None)

# Global instance
enhanced_verifier = EnhancedVerificationEngine()

def verify_with_enhanced_pipeline(text: str, use_openai: bool = True) -> Dict:
    """
    Main function to verify content using enhanced 8-step pipeline
    """
    return enhanced_verifier.verify_content(text, use_openai)
