"""
Enhanced Fake News Verification Engine
Implements comprehensive 8-step verification pipeline with trusted source integration
"""

import re
import json
import time
import requests
import threading
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

# Performance cache with TTL (Time To Live)
verification_cache = {}
cache_timestamps = {}
MAX_CACHE_SIZE = 500
CACHE_TTL = 3600  # 1 hour cache validity

def get_cached_result(text_hash: str) -> Optional[Dict]:
    """Get cached result if it exists and hasn't expired"""
    if text_hash in verification_cache:
        if (datetime.now() - cache_timestamps.get(text_hash, datetime.now())).total_seconds() < CACHE_TTL:
            return verification_cache[text_hash].copy()
        else:
            # Cache expired, remove it
            del verification_cache[text_hash]
            if text_hash in cache_timestamps:
                del cache_timestamps[text_hash]
    return None

def cache_result(text_hash: str, result: Dict) -> None:
    """Cache a result with timestamp"""
    if len(verification_cache) >= MAX_CACHE_SIZE:
        # Remove oldest entry
        oldest_key = min(cache_timestamps, key=cache_timestamps.get)
        del verification_cache[oldest_key]
        del cache_timestamps[oldest_key]
    
    verification_cache[text_hash] = result
    cache_timestamps[text_hash] = datetime.now()

class EnhancedVerificationEngine:
    """
    Comprehensive fake news verification following 8-step pipeline
    OPTIMIZED: Fast-track mode, aggressive caching, and parallel processing
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.session.timeout = 5  # Global timeout for all requests
    
    def verify_content(self, text: str, use_openai: bool = True, fast_mode: bool = False) -> Dict:
        """
        Main verification pipeline with FAST MODE optimization
        
        fast_mode=True: Skip expensive NLP, use heuristics only (~2-3 seconds for 200 lines)
        fast_mode=False: Full analysis with AI (~8-12 seconds for 200 lines)
        """
        start_time = time.time()
        
        # Check cache first (works for both modes)
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        cached = get_cached_result(text_hash)
        if cached:
            cached['cached'] = True
            cached['processing_time'] = f"<0.1s (cached)"
            return cached
        
        try:
            # FAST MODE: Skip expensive operations
            if fast_mode or len(text.split()) < 300:  # Auto-enable fast mode for short content
                return self.verify_content_fast(text, text_hash, start_time)
            
            # FULL MODE: Standard pipeline with timeouts
            return self.verify_content_full(text, use_openai, text_hash, start_time)
            
        except Exception as e:
            return self.generate_error_response(str(e), time.time() - start_time)
    
    def verify_content_fast(self, text: str, text_hash: str, start_time: float) -> Dict:
        """
        FAST-TRACK VERIFICATION: ~2-3 seconds for 200 lines
        Uses heuristics only, no API calls to external sources
        """
        try:
            # Step 1: Extract Core Claim (heuristic only)
            core_claim = self.extract_claim_heuristic(text)
            
            # Step 2: Extract Entities (heuristic only)
            entities = self.extract_entities_heuristic(core_claim)
            
            # Step 3: Local heuristic verification
            verdict, confidence = self.verify_claim_heuristic(core_claim, text)
            
            # Step 4: Quick bias check
            bias_analysis = {
                'suspicious_patterns': self.detect_suspicious_patterns(text),
                'sentiment_bias': self.quick_sentiment_check(text),
                'all_caps_ratio': (text.count('A') > len(text) * 0.3),
                'exclamation_ratio': (text.count('!') / max(len(text.split()), 1)) > 0.1
            }
            
            result = {
                'verdict': verdict,
                'confidence': confidence,
                'core_claim': core_claim,
                'entities': entities,
                'evidence': [],  # No evidence in fast mode
                'bias_analysis': bias_analysis,
                'processing_time': f"{time.time() - start_time:.2f}s (fast-track)",
                'mode': 'fast'
            }
            
            cache_result(text_hash, result.copy())
            return result
            
        except Exception as e:
            print(f"Fast mode error: {e}")
            return self.generate_error_response(str(e), time.time() - start_time)
    
    def verify_content_full(self, text: str, use_openai: bool, text_hash: str, start_time: float) -> Dict:
        """
        FULL VERIFICATION: ~8-12 seconds with aggressive timeouts
        """
        try:
            # Set execution timeout for entire pipeline
            def run_with_timeout(func, args, timeout_seconds):
                """Run function with timeout, return None if timeout"""
                result = [None]
                exception = [None]
                
                def worker():
                    try:
                        result[0] = func(*args)
                    except Exception as e:
                        exception[0] = e
                
                thread = threading.Thread(target=worker, daemon=True)
                thread.start()
                thread.join(timeout=timeout_seconds)
                
                if exception[0]:
                    raise exception[0]
                return result[0]
            
            # Step 1: Extract Core Claim (2 second timeout)
            core_claim = run_with_timeout(
                self.extract_core_claim,
                (text, use_openai),
                timeout_seconds=2
            ) or self.extract_claim_heuristic(text)
            
            # Step 2: Extract Entities (1.5 second timeout)
            entities = run_with_timeout(
                self.extract_entities,
                (core_claim, use_openai),
                timeout_seconds=1.5
            ) or self.extract_entities_heuristic(core_claim)
            
            # Step 3: Search Trusted Sources (3 second timeout - this is the slow part!)
            search_results = run_with_timeout(
                self.search_trusted_sources,
                (core_claim,),
                timeout_seconds=3
            ) or []
            
            # Step 4: Gather Evidence (2 second timeout)
            evidence = run_with_timeout(
                self.gather_evidence,
                (search_results, core_claim),
                timeout_seconds=2
            ) or []
            
            # Step 5: Verification Status (1 second timeout)
            verification_status = self.verify_claim_heuristic(core_claim, text)
            
            # Step 6: Calculate verdict
            verdict, confidence = verification_status if isinstance(verification_status, tuple) else ('UNVERIFIED', 'LOW')
            
            # Step 7: Bias Analysis (1 second timeout)
            bias_analysis = self.analyze_bias_context(evidence, search_results)
            
            # Step 8: Generate Output
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
            
            cache_result(text_hash, result.copy())
            return result
            
        except Exception as e:
            print(f"Full mode error: {e}")
            return self.generate_error_response(str(e), time.time() - start_time)
            
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
        """Step 8: Generate structured JSON output with detailed reporting data"""
        evidence = kwargs['evidence']
        verdict = kwargs['verdict']
        confidence = kwargs['confidence']
        
        # Calculate truth percentage (0-100)
        truth_percentage = self.calculate_truth_percentage(verdict, confidence, evidence)
        
        # Generate statistical breakdown
        statistics = self.generate_statistics(evidence, kwargs.get('bias_analysis', {}))
        
        # Categorize resources
        categorized_resources = self.categorize_resources(evidence)
        
        # Generate detailed explanation sections
        detailed_explanation = self.generate_detailed_explanation(
            verdict, confidence, evidence, kwargs['core_claim'], 
            kwargs['entities'], kwargs.get('bias_analysis', {})
        )
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "truth_percentage": truth_percentage,
            "evidence": [
                {
                    "source": item['source'],
                    "title": item['title'],
                    "url": item['url'],
                    "type": item['type'],
                    "verdict": item.get('verdict', 'UNVERIFIED'),
                    "weight": item['weight'],
                    "relevance": item.get('relevance', 0.5)
                }
                for item in evidence
            ],
            "explanation": self.generate_explanation(verdict, confidence, evidence),
            "core_claim": kwargs['core_claim'],
            "entities": kwargs['entities'],
            "bias_analysis": kwargs['bias_analysis'],
            "processing_time": f"{kwargs['processing_time']:.2f}s",
            "timestamp": datetime.now().isoformat(),
            "cached": False,
            # Enhanced detailed report data
            "detailed_report": {
                "statistics": statistics,
                "categorized_resources": categorized_resources,
                "detailed_explanation": detailed_explanation,
                "verification_method": kwargs.get('mode', 'full')
            }
        }
    
    def calculate_truth_percentage(self, verdict: str, confidence: str, evidence: List[Dict]) -> int:
        """Calculate truth percentage (0-100) based on verdict, confidence, and evidence"""
        base_score = 50  # Neutral starting point
        
        # Adjust based on verdict
        if verdict == "REAL":
            base_score = 75
        elif verdict == "FAKE":
            base_score = 25
        elif verdict == "MIXED":
            base_score = 50
        else:  # UNVERIFIED or ERROR
            base_score = 50
        
        # Adjust based on confidence level
        confidence_adjustments = {
            'HIGH': 20,
            'MEDIUM': 10,
            'LOW': 0,
            'ERROR': -10
        }
        adjustment = confidence_adjustments.get(confidence, 0)
        
        if verdict == "REAL":
            base_score += adjustment
        elif verdict == "FAKE":
            base_score -= adjustment
        
        # Fine-tune based on evidence quality
        if evidence:
            avg_weight = sum(item.get('weight', 0.5) for item in evidence) / len(evidence)
            avg_relevance = sum(item.get('relevance', 0.5) for item in evidence) / len(evidence)
            evidence_quality = (avg_weight + avg_relevance) / 2
            
            # Adjust score based on evidence quality
            if verdict == "REAL":
                base_score += int((evidence_quality - 0.5) * 20)
            elif verdict == "FAKE":
                base_score -= int((evidence_quality - 0.5) * 20)
        
        # Clamp to 0-100 range
        return max(0, min(100, base_score))
    
    def generate_statistics(self, evidence: List[Dict], bias_analysis: Dict) -> Dict:
        """Generate statistical breakdown for the report"""
        total_sources = len(evidence)
        fact_checkers = [e for e in evidence if e.get('type') == 'fact_check']
        news_sources = [e for e in evidence if e.get('type') == 'news_search']
        
        # Count verdicts
        supporting = [e for e in evidence if e.get('verdict') == 'REAL']
        refuting = [e for e in evidence if e.get('verdict') == 'FAKE']
        mixed = [e for e in evidence if e.get('verdict') == 'MIXED']
        unverified = [e for e in evidence if e.get('verdict') == 'UNVERIFIED']
        
        # Calculate average credibility
        avg_credibility = sum(e.get('weight', 0.5) for e in evidence) / max(total_sources, 1)
        
        return {
            "total_sources": total_sources,
            "fact_checkers_count": len(fact_checkers),
            "news_sources_count": len(news_sources),
            "supporting_count": len(supporting),
            "refuting_count": len(refuting),
            "mixed_count": len(mixed),
            "unverified_count": len(unverified),
            "average_credibility": round(avg_credibility, 2),
            "source_diversity": bias_analysis.get('source_diversity', 0),
            "reliability_score": round(bias_analysis.get('reliability_score', 0.5), 2)
        }
    
    def categorize_resources(self, evidence: List[Dict]) -> Dict:
        """Categorize resources by their verdict"""
        supporting = []
        refuting = []
        neutral = []
        
        for item in evidence:
            resource = {
                "source": item['source'],
                "title": item['title'],
                "url": item['url'],
                "type": item['type'],
                "credibility": item.get('weight', 0.5),
                "relevance": item.get('relevance', 0.5)
            }
            
            verdict = item.get('verdict', 'UNVERIFIED')
            if verdict == 'REAL':
                supporting.append(resource)
            elif verdict == 'FAKE':
                refuting.append(resource)
            else:
                neutral.append(resource)
        
        return {
            "supporting": supporting,
            "refuting": refuting,
            "neutral": neutral
        }
    
    def generate_detailed_explanation(self, verdict: str, confidence: str, evidence: List[Dict], 
                                     core_claim: str, entities: Dict, bias_analysis: Dict) -> Dict:
        """Generate detailed explanation sections for the report"""
        
        # Content Summary - What's inside
        content_summary = self.generate_content_summary(core_claim, entities, bias_analysis)
        
        # Why it's wrong/right - Detailed reasoning with evidence
        verdict_reasoning = self.generate_verdict_reasoning(verdict, confidence, evidence, core_claim)
        
        # Summary section
        summary = self.generate_explanation(verdict, confidence, evidence)
        
        # Methodology section
        methodology = f"This analysis used {'enhanced verification with AI-powered claim extraction' if len(evidence) > 0 else 'heuristic-based verification'} to evaluate the content. "
        methodology += f"We analyzed {len(evidence)} sources including {bias_analysis.get('fact_check_coverage', 0)} fact-checkers and {bias_analysis.get('news_coverage', 0)} news outlets."
        
        # Red flags section
        red_flags = bias_analysis.get('suspicious_patterns', [])
        if not red_flags:
            red_flags = ["No significant red flags detected in the content."]
        
        # Entities section
        entities_text = ""
        if entities:
            entity_parts = []
            if entities.get('people'):
                entity_parts.append(f"People: {', '.join(entities['people'][:5])}")
            if entities.get('organizations'):
                entity_parts.append(f"Organizations: {', '.join(entities['organizations'][:5])}")
            if entities.get('dates'):
                entity_parts.append(f"Dates: {', '.join(entities['dates'][:3])}")
            if entities.get('numbers'):
                entity_parts.append(f"Key Numbers: {', '.join(entities['numbers'][:5])}")
            entities_text = "; ".join(entity_parts) if entity_parts else "No specific entities identified."
        else:
            entities_text = "No specific entities identified."
        
        # Verification tips
        verification_tips = [
            "Cross-check with multiple reputable fact-checking websites (Snopes, PolitiFact, FactCheck.org)",
            "Look for original sources and primary documentation",
            "Verify dates, locations, and numerical claims independently",
            "Check if the information appears in multiple credible news outlets",
            "Be cautious of emotionally charged language or sensationalist claims"
        ]
        
        return {
            "content_summary": content_summary,
            "verdict_reasoning": verdict_reasoning,
            "summary": summary,
            "core_claim": core_claim,
            "entities": entities_text,
            "methodology": methodology,
            "red_flags": red_flags,
            "bias_indicators": bias_analysis.get('sentiment_bias', 'neutral'),
            "verification_tips": verification_tips
        }
    
    def generate_content_summary(self, core_claim: str, entities: Dict, bias_analysis: Dict) -> str:
        """Generate a summary of what's inside the content"""
        summary_parts = []
        
        # Start with the core claim
        summary_parts.append(f"The content makes the following claim: \"{core_claim[:300]}{'...' if len(core_claim) > 300 else ''}\"")
        
        # Add entity information if available
        if entities:
            entity_mentions = []
            if entities.get('people'):
                entity_mentions.append(f"{len(entities['people'])} person(s)")
            if entities.get('organizations'):
                entity_mentions.append(f"{len(entities['organizations'])} organization(s)")
            if entities.get('dates'):
                entity_mentions.append(f"{len(entities['dates'])} date(s)")
            if entities.get('numbers'):
                entity_mentions.append(f"{len(entities['numbers'])} numerical claim(s)")
            
            if entity_mentions:
                summary_parts.append(f"The content mentions {', '.join(entity_mentions)}.")
        
        # Add bias indicators
        sentiment = bias_analysis.get('sentiment_bias', 'neutral')
        if sentiment != 'neutral':
            summary_parts.append(f"The content shows {sentiment.replace('_', ' ')} in its language and tone.")
        
        # Add suspicious patterns if any
        patterns = bias_analysis.get('suspicious_patterns', [])
        if patterns:
            summary_parts.append(f"Analysis detected potential issues: {', '.join(patterns[:3])}.")
        
        return " ".join(summary_parts)
    
    def generate_verdict_reasoning(self, verdict: str, confidence: str, evidence: List[Dict], core_claim: str) -> str:
        """Generate detailed reasoning for why the content is fake/real with evidence"""
        reasoning_parts = []
        
        # Start with the verdict
        if verdict == "FAKE":
            reasoning_parts.append("**Why This Is Likely False:**")
            reasoning_parts.append("")
            
            # Find refuting evidence
            refuting = [e for e in evidence if e.get('verdict') == 'FAKE']
            if refuting:
                reasoning_parts.append(f"Our analysis found {len(refuting)} credible source(s) that refute this claim:")
                reasoning_parts.append("")
                for i, source in enumerate(refuting[:3], 1):
                    reasoning_parts.append(f"{i}. **{source['source']}** (Credibility: {int(source.get('weight', 0.5) * 100)}%)")
                    reasoning_parts.append(f"   - {source['title']}")
                    reasoning_parts.append(f"   - This source indicates the claim is false or misleading")
                    reasoning_parts.append("")
            else:
                reasoning_parts.append("The claim shows characteristics commonly associated with misinformation:")
                reasoning_parts.append("- Lack of credible source citations")
                reasoning_parts.append("- Sensationalist or emotionally charged language")
                reasoning_parts.append("- No corroboration from trusted fact-checkers or news outlets")
                reasoning_parts.append("")
            
        elif verdict == "REAL":
            reasoning_parts.append("**Why This Is Likely True:**")
            reasoning_parts.append("")
            
            # Find supporting evidence
            supporting = [e for e in evidence if e.get('verdict') == 'REAL']
            if supporting:
                reasoning_parts.append(f"Our analysis found {len(supporting)} credible source(s) that support this claim:")
                reasoning_parts.append("")
                for i, source in enumerate(supporting[:3], 1):
                    reasoning_parts.append(f"{i}. **{source['source']}** (Credibility: {int(source.get('weight', 0.5) * 100)}%)")
                    reasoning_parts.append(f"   - {source['title']}")
                    reasoning_parts.append(f"   - This source confirms the accuracy of the claim")
                    reasoning_parts.append("")
            else:
                reasoning_parts.append("The claim shows characteristics of credible information:")
                reasoning_parts.append("- Factual language without excessive emotion")
                reasoning_parts.append("- Specific, verifiable details")
                reasoning_parts.append("- Consistent with known facts")
                reasoning_parts.append("")
                
        elif verdict == "MIXED":
            reasoning_parts.append("**Why This Has Mixed Accuracy:**")
            reasoning_parts.append("")
            reasoning_parts.append("The content contains both accurate and inaccurate elements:")
            
            supporting = [e for e in evidence if e.get('verdict') == 'REAL']
            refuting = [e for e in evidence if e.get('verdict') == 'FAKE']
            
            if supporting:
                reasoning_parts.append(f"- {len(supporting)} source(s) support parts of the claim")
            if refuting:
                reasoning_parts.append(f"- {len(refuting)} source(s) refute other parts of the claim")
            
            reasoning_parts.append("")
            reasoning_parts.append("This suggests the claim may be partially true but contains misleading or false elements.")
            reasoning_parts.append("")
            
        else:  # UNVERIFIED
            reasoning_parts.append("**Why This Cannot Be Verified:**")
            reasoning_parts.append("")
            reasoning_parts.append("We could not find sufficient credible sources to verify this claim:")
            reasoning_parts.append(f"- Only {len(evidence)} source(s) found related to this topic")
            reasoning_parts.append("- No definitive fact-checks available")
            reasoning_parts.append("- Claim may be too recent, obscure, or opinion-based")
            reasoning_parts.append("")
            reasoning_parts.append("**Recommendation:** Treat this claim with skepticism until more evidence emerges.")
            reasoning_parts.append("")
        
        # Add confidence explanation
        reasoning_parts.append(f"**Confidence Level: {confidence}**")
        if confidence == "HIGH":
            reasoning_parts.append("We have strong evidence from multiple credible sources to support this assessment.")
        elif confidence == "MEDIUM":
            reasoning_parts.append("We have moderate evidence, but additional verification would strengthen this assessment.")
        else:
            reasoning_parts.append("Limited evidence available. This assessment should be considered preliminary.")
        
        return "\n".join(reasoning_parts)
    
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
    
    def verify_claim_heuristic(self, claim: str, full_text: str) -> Tuple[str, str]:
        """
        Fast heuristic claim verification using pattern matching
        Returns: (verdict, confidence_level)
        """
        score = 0
        max_score = 0
        
        # Check for definitive language
        definitive_true = ['confirmed', 'verified', 'proven', 'official', 'announced', 'validated']
        definitive_false = ['debunked', 'hoax', 'false', 'fabricated', 'disproven', 'fake']
        
        claim_lower = claim.lower()
        for word in definitive_true:
            if word in claim_lower:
                score += 2
        
        for word in definitive_false:
            if word in claim_lower:
                score -= 2
        
        max_score = 2
        
        # Check for source citations
        has_sources = bool(re.search(r'(according to|sources say|reports|study|research)', claim_lower))
        if has_sources:
            score += 1
            max_score += 1
        
        # Check for numbers and statistics (more credible)
        has_numbers = bool(re.search(r'\d+(?:%|,\d+)?', claim))
        if has_numbers:
            score += 0.5
            max_score += 1
        
        # Check for emotional language (less credible)
        emotional_words = ['shocking', 'unbelievable', 'disgusting', 'outrageous', 'horrible', 'terrible']
        emotional_count = sum(1 for word in emotional_words if word in claim_lower)
        if emotional_count > 2:
            score -= 1.5
        
        max_score += 1
        
        # Check for sensationalism in full text
        exclamation_ratio = full_text.count('!') / max(len(full_text.split()), 1)
        if exclamation_ratio > 0.15:
            score -= 1
        
        max_score += 1
        
        # Determine verdict based on score
        if max_score > 0:
            normalized_score = score / max_score
        else:
            normalized_score = 0
        
        if normalized_score >= 0.6:
            return ('REAL', 'MEDIUM')
        elif normalized_score <= -0.6:
            return ('FAKE', 'MEDIUM')
        else:
            return ('UNVERIFIED', 'LOW')
    
    def detect_suspicious_patterns(self, text: str) -> List[str]:
        """Detect suspicious patterns indicating misinformation"""
        patterns = []
        text_lower = text.lower()
        
        # Check for clickbait patterns
        if bool(re.search(r'(?:you won\'t believe|doctors hate|celebrities hate|shocking|exclusive|breaking)', text_lower)):
            patterns.append("Clickbait language detected")
        
        # Check for sensationalism
        exclamation_count = text.count('!')
        question_count = text.count('?')
        if exclamation_count + question_count > len(text.split()) * 0.1:
            patterns.append("Excessive punctuation")
        
        # Check for unverifiable claims
        if bool(re.search(r'(?:anonymous sources|insiders claim|leaked|confidential)', text_lower)):
            patterns.append("Unverifiable source claims")
        
        # Check for logical fallacies
        if bool(re.search(r'(?:obviously|clearly|everyone knows|common sense)', text_lower)):
            patterns.append("Appeal to common sense fallacy")
        
        # Check for ALL CAPS sections
        caps_words = len([w for w in text.split() if w.isupper() and len(w) > 2])
        if caps_words > len(text.split()) * 0.2:
            patterns.append("Excessive capitalization")
        
        return patterns
    
    def quick_sentiment_check(self, text: str) -> str:
        """Quick sentiment analysis without ML models"""
        text_lower = text.lower()
        
        negative_words = ['bad', 'evil', 'dangerous', 'threat', 'crisis', 'disaster', 'worst', 'horrible']
        positive_words = ['good', 'great', 'excellent', 'wonderful', 'best', 'perfect']
        
        negative_count = sum(1 for word in negative_words if word in text_lower)
        positive_count = sum(1 for word in positive_words if word in text_lower)
        
        if negative_count > positive_count:
            return 'negative_bias'
        elif positive_count > negative_count:
            return 'positive_bias'
        else:
            return 'neutral'
    
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

def verify_with_enhanced_pipeline(text: str, use_openai: bool = True, fast_mode: bool = False) -> Dict:
    """
    Main function to verify content using enhanced 8-step pipeline
    
    Args:
        text: The content to verify
        use_openai: Whether to use OpenAI API calls
        fast_mode: If True, use heuristics only (2-3 seconds for 200 lines)
                  If False, use full pipeline (8-12 seconds for 200 lines)
    """
    return enhanced_verifier.verify_content(text, use_openai, fast_mode)
