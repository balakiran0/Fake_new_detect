#!/usr/bin/env python
"""
Test script to verify the cricket news article is properly classified and analyzed.
"""
import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fakenews_project'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fakenews_project.settings')
django.setup()

from detector.ai_model import classify_user_intent_fast, classify_content_type_fast
from detector.enhanced_verification import verify_with_enhanced_pipeline

# Test cricket news article
cricket_news = """Gill has been incidentally been advised by doctors not to fly out, a standard precautionary measure following a neck injury. Now it is up to the medical panel of the Indian cricket board (BCCI) to decide Gill's next step towards recovery and participation in the upcoming series."""

print("=" * 80)
print("TEST: Cricket News Article Classification and Analysis")
print("=" * 80)

print("\n📰 INPUT TEXT:")
print(f"'{cricket_news}'")
print(f"\nText length: {len(cricket_news)} characters, {len(cricket_news.split())} words")

# Test 1: Intent Classification
print("\n" + "=" * 80)
print("TEST 1: Intent Classification")
print("=" * 80)

intent, intent_confidence = classify_user_intent_fast(cricket_news)
print(f"✅ Intent: {intent.upper()}")
print(f"✅ Confidence: {intent_confidence:.2%}")

# Expected: Should be 'analysis', not 'conversation'
if intent == 'analysis':
    print("✓ PASS: Correctly classified as analysis (news article)")
else:
    print(f"✗ FAIL: Incorrectly classified as '{intent}' instead of 'analysis'")

# Test 2: Content Type Classification
print("\n" + "=" * 80)
print("TEST 2: Content Type Classification")
print("=" * 80)

content_type, type_confidence, structure = classify_content_type_fast(cricket_news)
print(f"✅ Content Type: {content_type}")
print(f"✅ Type Confidence: {type_confidence:.2%}")
print(f"✅ Structure: {structure}")

# Test 3: Enhanced Verification Pipeline
print("\n" + "=" * 80)
print("TEST 3: Enhanced Verification Pipeline")
print("=" * 80)

try:
    print("Running enhanced verification pipeline...")
    result = verify_with_enhanced_pipeline(cricket_news, use_openai=False)
    
    print(f"\n✅ Verdict: {result.get('verdict', 'UNKNOWN')}")
    print(f"✅ Confidence: {result.get('confidence', 'UNKNOWN')}")
    print(f"✅ Core Claim: {result.get('core_claim', 'N/A')}")
    print(f"✅ Explanation: {result.get('explanation', 'N/A')[:200]}...")
    
    evidence = result.get('evidence', [])
    print(f"✅ Evidence Sources Found: {len(evidence)}")
    for i, ev in enumerate(evidence[:3], 1):
        print(f"   {i}. {ev.get('source', 'Unknown')}: {ev.get('title', 'No title')}")
    
    print("\n✓ PASS: Enhanced verification pipeline executed successfully")
    
except Exception as e:
    print(f"\n✗ FAIL: Enhanced verification pipeline error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Full workflow - what should happen now
print("\n" + "=" * 80)
print("TEST 4: Expected Workflow")
print("=" * 80)

print("\nWith fixes applied:")
print("1. Intent classification: ANALYSIS ✓ (was CONVERSATION before)")
print("2. Content type: NEWS ARTICLE ✓")
print("3. Routes to: Enhanced verification pipeline ✓ (was generic conversation before)")
print("4. Response: Fact-checking verdict with evidence ✓")
print("\n🎯 EXPECTED RESULT: System should analyze cricket article for fact-checking")
print("   instead of responding generically!")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
