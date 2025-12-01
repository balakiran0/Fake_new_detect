from rest_framework.views import APIView
from rest_framework.response import Response
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from datetime import timedelta
import time
from .models import AnalysisResult
from .ai_model import (
    classify_text_type,
    detect_fake_news,
    generate_community_notes,
    classify_user_intent_fast,
    classify_content_type_fast,
    analyze_content_structure_fast,
    HF_AVAILABLE,
    _initialize_pipelines,
    LAST_ERROR,
    OPENAI_AVAILABLE,
    OPENAI_API_KEY
)
from .ai_model_enhanced import IntelligentConversationEngine
from .enhanced_verification import verify_with_enhanced_pipeline
import base64
from io import BytesIO
try:
    from PIL import Image  # optional
except Exception:
    Image = None
try:
    import pytesseract  # optional
except Exception:
    pytesseract = None
try:
    from pypdf import PdfReader  # optional
except Exception:
    PdfReader = None
import re
import requests
from bs4 import BeautifulSoup
import tempfile
import os

# Initialize the enhanced conversation engine once at module load
_conversation_engine = None

def get_conversation_engine():
    """Lazily initialize the conversation engine"""
    global _conversation_engine
    if _conversation_engine is None:
        try:
            _conversation_engine = IntelligentConversationEngine()
        except Exception as e:
            print(f"Warning: Failed to initialize IntelligentConversationEngine: {e}")
            _conversation_engine = False  # Mark as failed
    return _conversation_engine if _conversation_engine is not False else None


@method_decorator(csrf_exempt, name='dispatch')
class FakeNewsDetectorAPIView(APIView):
    def post(self, request):
        # Start performance timer
        start_time = time.time()
        
        # Removed _initialize_pipelines() call for faster performance

        input_data = request.data.get('text', '').strip()
        files = request.data.get('files') or []
        
        # Fast content analysis - classify both intent and content type
        intent, intent_confidence = classify_user_intent_fast(input_data)
        content_type, type_confidence, structure_analysis = classify_content_type_fast(input_data)
        
        # If it's conversational and no files uploaded, respond conversationally
        if intent == 'conversation' and len(files) == 0 and intent_confidence > 0.6:
            # Try to get conversation history from request
            conversation_history = request.data.get('conversation_history', [])
            
            # Use enhanced conversation engine if available
            engine = get_conversation_engine()
            if engine:
                try:
                    conversational_response = engine.understand_and_respond(input_data, conversation_history)
                except Exception as e:
                    print(f"Enhanced engine error: {e}, falling back to basic response")
                    conversational_response = f"I understand: {input_data[:100]}... Let me help with that!"
            else:
                conversational_response = f"I understand: {input_data[:100]}... Let me help with that!"
            
            # Calculate response time
            response_time = round(time.time() - start_time, 2)
            
            return Response({
                "intent": "conversation",
                "response": conversational_response,
                "confidence": intent_confidence,
                "content_analysis": {
                    "detected_type": content_type,
                    "type_confidence": type_confidence,
                    "structure": structure_analysis
                },
                "powered_by": "GPT" if (OPENAI_AVAILABLE or OPENAI_API_KEY) else "Local AI",
                "processing_time": f"{response_time}s (optimized)"
            })

        # If user sent a URL, fetch and extract text
        url_pattern = re.compile(r'^https?://\S+$')
        page_title = None
        is_url_source = False
        origin_url = None
        if url_pattern.match(input_data):
            try:
                response = requests.get(input_data, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)  # Reduced timeout
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                is_url_source = True
                origin_url = input_data
                # capture title and metadata for heuristics
                if soup.title and soup.title.string:
                    page_title = soup.title.string.strip()
                og_type = soup.find('meta', attrs={'property': 'og:type'})
                article_tag = soup.find(['article'])
                # remove script/style
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                input_data = '\n'.join(chunk for chunk in chunks if chunk)
                # store a hint for downstream classification
                request._fetched_page_meta = {
                    'og_type': (og_type.get('content').lower() if og_type and og_type.get('content') else ''),
                    'has_article_tag': bool(article_tag),
                    'title': page_title or ''
                }
            except requests.RequestException as e:
                return Response({"error": f"Failed to fetch content from URL: {str(e)}"}, status=400)

        # Extract text from uploaded files (PDF, images via OCR, video via Whisper)
        extracted_text = ""
        filenames = []
        for f in files:
            try:
                header, encoded = f.get('data').split(",", 1)
                file_bytes = base64.b64decode(encoded)
                file_type = f.get('type', '')
                if f.get('filename'):
                    filenames.append(f.get('filename'))

                if file_type == 'application/pdf':
                    if PdfReader is None:
                        extracted_text += f"\n[PDF support not available: pypdf not installed]"
                    else:
                        reader = PdfReader(BytesIO(file_bytes))
                        for page in reader.pages:
                            extracted_text += (page.extract_text() or "") + "\n"
                elif file_type.startswith('image/'):
                    if Image is None or pytesseract is None:
                        missing = []
                        if Image is None:
                            missing.append("Pillow")
                        if pytesseract is None:
                            missing.append("pytesseract")
                        extracted_text += f"\n[Image OCR not available: missing {' & '.join(missing)}]"
                    else:
                        image = Image.open(BytesIO(file_bytes))
                        try:
                            extracted_text += (pytesseract.image_to_string(image) or "") + "\n"
                        except Exception as e:
                            extracted_text += f"\n[Error running OCR: {str(e)}]"
                elif file_type.startswith('video/'):
                    # Transcribe audio track using Whisper (tiny model for speed)
                    try:
                        import whisper
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + (f.get('filename','').split('.')[-1] or 'mp4')) as tmp:
                            tmp.write(file_bytes)
                            tmp_path = tmp.name
                        try:
                            model = whisper.load_model("tiny")
                            result = model.transcribe(tmp_path)
                            transcript = result.get('text') or ''
                            if transcript:
                                extracted_text += transcript + "\n"
                        finally:
                            try:
                                os.remove(tmp_path)
                            except Exception:
                                pass
                    except Exception as e:
                        extracted_text += f"\n[Error transcribing video {f.get('filename','unknown')}: {str(e)}]"
            except Exception as e:
                extracted_text += f"\n[Error processing file {f.get('filename', 'unknown')}: {str(e)}]"

        full_text = (extracted_text + "\n" + input_data).strip()

        if not full_text:
            return Response({"error": "No text provided. Send JSON with key 'text' or upload a file."}, status=400)

        # Step 1: classify content type
        content_type, type_confidence = classify_text_type(full_text)
        # Heuristic override for URLs that look like articles
        meta = getattr(request, '_fetched_page_meta', None)
        if meta:
            long_enough = len(full_text.split()) > 150
            looks_article = (meta.get('og_type') == 'article') or meta.get('has_article_tag') or long_enough
            if looks_article:
                content_type, type_confidence = 'news article', max(type_confidence, 0.9)
        else:
            # Heuristic for pasted text that looks like an article
            lines = [l.strip() for l in full_text.splitlines() if l.strip()]
            first = lines[0] if lines else ''
            has_date = bool(re.search(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})\b', full_text))
            has_byline = bool(re.search(r'\bBy\s+[A-Z][a-z]+\b|Byline:', full_text))
            has_quotes = full_text.count('“') + full_text.count('"') >= 2
            is_headliney = len(first.split()) <= 16 and first.endswith(('!', '.', ''))
            if (has_date or has_byline or has_quotes or is_headliney) and len(full_text.split()) > 120:
                content_type, type_confidence = 'news article', max(type_confidence, 0.85)
        # Proceed with analysis for any content type (do not block); include type in response

        # Step 2: Choose verification method based on content length and type
        use_enhanced = len(full_text.split()) > 50 or content_type == 'news article'
        
        if use_enhanced:
            # Determine if we can use fast-track mode (heuristics only, no API calls)
            # Fast mode: <500 words OR no file uploads (simple paste)
            use_fast_mode = (len(full_text.split()) < 500 and len(files) == 0) or 'fast_mode' in request.data
            
            # Use enhanced 8-step verification pipeline with timeout protection
            try:
                enhanced_result = verify_with_enhanced_pipeline(
                    full_text, 
                    use_openai=OPENAI_AVAILABLE and not use_fast_mode,
                    fast_mode=use_fast_mode
                )
            except Exception as e:
                # Fallback to fast mode if something goes wrong
                print(f"Enhanced verification error, falling back to fast mode: {e}")
                enhanced_result = verify_with_enhanced_pipeline(
                    full_text, 
                    use_openai=False,
                    fast_mode=True
                )
            
            verdict = enhanced_result['verdict']
            confidence_level = enhanced_result['confidence']
            
            # Convert confidence level to score
            confidence_map = {'HIGH': 0.9, 'MEDIUM': 0.7, 'LOW': 0.5, 'ERROR': 0.3}
            confidence_score = confidence_map.get(confidence_level, 0.5)
            
            # Format enhanced response
            reply_text = f"Enhanced Verification Result: {verdict}\nConfidence: {confidence_level}\n\nCore Claim: {enhanced_result.get('core_claim', 'N/A')}\n\nExplanation: {enhanced_result.get('explanation', '')}"
            
            if enhanced_result.get('evidence'):
                reply_text += "\n\nEvidence Sources:\n"
                for evidence in enhanced_result['evidence'][:5]:  # Show top 5
                    reply_text += f"- {evidence['source']}: {evidence['title']} ({evidence['url']})\n"
            
            # Add bias analysis if available
            bias_analysis = enhanced_result.get('bias_analysis', {})
            if bias_analysis.get('suspicious_patterns'):
                reply_text += f"\n⚠️ Analysis Notes: {', '.join(bias_analysis['suspicious_patterns'])}"
            
            notes_obj = {
                'summary': enhanced_result.get('explanation', ''),
                'sources': [{
                    'site': e['source'],
                    'title': e['title'],
                    'url': e['url']
                } for e in enhanced_result.get('evidence', [])]
            }
        else:
            # Use original fast detection for shorter content
            verdict, confidence_score = detect_fake_news(full_text)
            search_text = (meta.get('title') if meta and meta.get('title') else full_text)
            notes_obj = generate_community_notes(search_text, verdict)
            confidence_pct = int(round(confidence_score * 100))
            reply_text = f"Quick Analysis: {verdict}\nConfidence: {confidence_pct}%\n\nExplanation: {notes_obj.get('summary','')}"
            if notes_obj.get('sources'):
                reply_text += "\n\nSources:\n" + "\n".join(
                    [f"- {s.get('site', 'Source')}: {s.get('title','')} ({s.get('url','')})" for s in notes_obj['sources']]
                )

            if confidence_score < 0.6:
                reply_text += "\nI don't have high confidence in this prediction. You may want to seek further verification."

        # Save result
        result_obj = AnalysisResult.objects.create(
            input_text=full_text,
            is_fake=(verdict == 'Fake'),
            confidence_score=confidence_score,
            generated_notes=notes_obj.get('summary')
        )

        # Calculate total response time
        total_response_time = round(time.time() - start_time, 2)
        
        # Prepare extracted content info
        extracted_content_info = {
            "source_type": "url" if origin_url else ("file" if filenames else "text"),
            "page_title": page_title if origin_url else None,
            "url": origin_url,
            "files": filenames,
            "extracted_text": full_text[:1000] + "..." if len(full_text) > 1000 else full_text,  # First 1000 chars
            "total_length": len(full_text),
            "word_count": len(full_text.split())
        }
        
        # Prepare response with enhanced data if available
        response_data = {
            "intent": "analysis",
            "verdict": verdict,
            "confidence": confidence_score,
            "notes": notes_obj,
            "reply": reply_text,
            "content_type": content_type,
            "content_type_confidence": type_confidence,
            "content_analysis": {
                "detected_type": content_type,
                "type_confidence": type_confidence,
                "structure": structure_analysis,
                "analysis_summary": f"Detected as {content_type.replace('_', ' ').title()} with {int(type_confidence*100)}% confidence"
            },
            "origin": {"url": origin_url, "files": filenames},
            "extracted_content": extracted_content_info,
            "id": result_obj.id,
            "processing_time": f"{total_response_time}s (optimized)",
            "verification_method": "enhanced" if use_enhanced else "standard"
        }
        
        # Add enhanced verification data if available
        if use_enhanced and 'enhanced_result' in locals():
            response_data["truth_percentage"] = enhanced_result.get('truth_percentage', 50)
            response_data["detailed_report"] = enhanced_result.get('detailed_report', {})
            response_data["enhanced_verification"] = {
                "core_claim": enhanced_result.get('core_claim', ''),
                "entities": enhanced_result.get('entities', {}),
                "evidence_count": len(enhanced_result.get('evidence', [])),
                "bias_analysis": enhanced_result.get('bias_analysis', {}),
                "confidence_level": enhanced_result.get('confidence', 'LOW'),
                "cached": enhanced_result.get('cached', False)
            }
        else:
            # For standard verification, calculate basic truth percentage
            truth_pct = int(confidence_score * 100) if verdict == 'Real' else int((1 - confidence_score) * 100)
            response_data["truth_percentage"] = truth_pct
        
        return Response(response_data)


@method_decorator(csrf_exempt, name='dispatch')
class RecentAnalysesAPIView(APIView):
    def get(self, request):
        latest = AnalysisResult.objects.order_by('-created_at')[:20]
        data = []
        for r in latest:
            data.append({
                'id': r.id,
                'created_at': r.created_at.isoformat(),
                'verdict': 'Fake' if r.is_fake else 'Real',
                'confidence': r.confidence_score,
                'snippet': r.input_text[:200],
                'notes': r.generated_notes or ''
            })
        return Response({"results": data}, status=200)


@method_decorator(csrf_exempt, name='dispatch')
class AnalysisDetailAPIView(APIView):
    def get(self, request, analysis_id):
        try:
            analysis = AnalysisResult.objects.get(id=analysis_id)
            
            # Format the detailed response
            data = {
                'id': analysis.id,
                'input_text': analysis.input_text,
                'verdict': 'Fake' if analysis.is_fake else 'Real',
                'confidence': analysis.confidence_score,
                'confidence_percentage': int(round(analysis.confidence_score * 100)),
                'notes': analysis.generated_notes or '',
                'created_at': analysis.created_at.isoformat(),
                'created_at_formatted': analysis.created_at.strftime('%B %d, %Y at %I:%M %p'),
                'snippet': analysis.input_text[:200] + ('...' if len(analysis.input_text) > 200 else ''),
                'word_count': len(analysis.input_text.split()),
                'character_count': len(analysis.input_text),
            }
            
            return Response(data, status=200)
            
        except AnalysisResult.DoesNotExist:
            return Response({"error": "Analysis not found"}, status=404)


@method_decorator(csrf_exempt, name='dispatch')
class EnhancedVerificationAPIView(APIView):
    def post(self, request):
        """Test endpoint for enhanced verification pipeline"""
        start_time = time.time()
        
        input_text = request.data.get('text', '').strip()
        if not input_text:
            return Response({"error": "No text provided"}, status=400)
        
        try:
            # Run enhanced verification
            result = verify_with_enhanced_pipeline(input_text, use_openai=OPENAI_AVAILABLE)
            
            # Add processing time
            result['total_processing_time'] = f"{time.time() - start_time:.2f}s"
            result['openai_available'] = OPENAI_AVAILABLE
            
            return Response(result, status=200)
            
        except Exception as e:
            return Response({
                "error": f"Enhanced verification failed: {str(e)}",
                "processing_time": f"{time.time() - start_time:.2f}s"
            }, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class TrendDataAPIView(APIView):
    def get(self, request):
        # Build last 7 days histogram of real/fake counts
        now = timezone.now()
        start = (now - timedelta(days=6)).date()
        buckets = {}
        for i in range(7):
            day = (start + timedelta(days=i)).isoformat()
            buckets[day] = {"real": 0, "fake": 0}
        qs = AnalysisResult.objects.filter(created_at__date__gte=start)
        for r in qs:
            day = r.created_at.date().isoformat()
            if day in buckets:
                if r.is_fake:
                    buckets[day]["fake"] += 1
                else:
                    buckets[day]["real"] += 1
        return Response(buckets, status=200)
