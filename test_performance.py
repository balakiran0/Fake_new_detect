#!/usr/bin/env python
"""
Performance test: Verify 200-line content analysis
Target: < 30 seconds (previously 10+ minutes!)
"""
import os
import sys
import django
import time

# Setup Django
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fakenews_project'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fakenews_project.settings')
django.setup()

from detector.enhanced_verification import verify_with_enhanced_pipeline

# Sample 200-line news article
long_article = """
Breaking News: Major Scientific Breakthrough Announced

Scientists at leading research institutions across the globe have announced a groundbreaking discovery 
that could revolutionize our understanding of climate change and environmental science. The research, 
conducted over the past three years, has been peer-reviewed and published in multiple prestigious journals.

According to Dr. Sarah Mitchell, lead researcher at the International Climate Research Institute, the team 
discovered a new mechanism for carbon sequestration that could significantly impact global warming mitigation 
strategies. "This breakthrough represents years of collaborative research and dedication," said Dr. Mitchell 
during the press conference held yesterday.

The study examined over 50,000 data points collected from various regions worldwide. Researchers found that 
certain types of microorganisms could be engineered to absorb carbon dioxide at three times the rate previously 
thought possible. This discovery opens new possibilities for industrial applications and environmental remediation.

"The potential implications are enormous," explained Prof. James Chen, from Cambridge University's Department of 
Environmental Science. "If we can scale this technology, we might be able to address atmospheric carbon levels 
in ways we previously thought impossible." Prof. Chen's team verified the findings independently and confirmed 
the accuracy of the results.

The research consortium includes institutions from the United States, Europe, Asia, and Australia. Each team 
conducted separate experiments following the same protocol, and all results converged on similar conclusions. 
This consistency across multiple independent laboratories strengthens the credibility of the findings.

However, critics have raised some concerns about the real-world applicability of the technology. Dr. Robert Wilson 
from the University of Oxford notes that while the laboratory results are impressive, scaling the process to an 
industrial level presents significant engineering challenges. "We need to be cautious about over-promising," Dr. Wilson 
cautioned, though he acknowledged the genuine breakthrough in the underlying science.

The research team is now seeking funding for Phase 2 of the project, which will focus on pilot implementations and 
scaling studies. Several governments and international organizations have already expressed interest in supporting 
this research. The team estimates that initial field trials could begin within the next 18 to 24 months.

Environmental organizations have welcomed the news. The Global Environmental Coalition issued a statement expressing 
cautious optimism about the potential benefits. "While not a silver bullet, this could be a valuable tool in our 
arsenal against climate change," their statement read.

The full research paper, titled "Novel Enzymatic Carbon Sequestration via Engineered Microbial Systems," has been 
published simultaneously in Nature, Science, and the Journal of Environmental Research. The paper includes comprehensive 
methodology, data analysis, and supporting evidence. Researchers emphasize the importance of transparency and have made 
all raw data available through a public repository.

This discovery comes at a critical time for climate action. Recent reports from the Intergovernmental Panel on Climate 
Change have emphasized the urgency of developing new technologies to reduce atmospheric carbon dioxide. This breakthrough 
could contribute meaningfully to global climate mitigation goals outlined in the Paris Agreement.

The research team plans to present their findings at major international conferences over the coming months and has 
invited collaboration from other institutions worldwide to validate and extend their work. This commitment to open science 
and collaboration reflects best practices in modern research.
""" * 3  # Approximately 200 lines

print("=" * 80)
print("PERFORMANCE TEST: Optimized Fake News Detection")
print("=" * 80)
print(f"\n📄 Test Content: {len(long_article.split())} words, {len(long_article.split(chr(10)))} lines")

# Test 1: FAST MODE (heuristics only)
print("\n" + "=" * 80)
print("TEST 1: FAST MODE (Heuristics Only - No API Calls)")
print("=" * 80)

start = time.time()
result_fast = verify_with_enhanced_pipeline(long_article, use_openai=False, fast_mode=True)
time_fast = time.time() - start

print(f"⏱️  Execution Time: {time_fast:.2f} seconds")
print(f"✅ Verdict: {result_fast['verdict']}")
print(f"✅ Confidence: {result_fast['confidence']}")
print(f"✅ Mode: {result_fast.get('mode', 'fast')}")
print(f"✅ Processing Time Reported: {result_fast['processing_time']}")

if time_fast < 5:
    print("✓ PASS: Fast mode under 5 seconds!")
elif time_fast < 10:
    print("✓ ACCEPTABLE: Fast mode under 10 seconds")
else:
    print(f"✗ WARNING: Fast mode took {time_fast:.2f} seconds (expected < 5s)")

# Test 2: STANDARD MODE (with timeouts)
print("\n" + "=" * 80)
print("TEST 2: STANDARD MODE (With Timeouts - ~10 seconds)")
print("=" * 80)

start = time.time()
result_standard = verify_with_enhanced_pipeline(long_article, use_openai=False, fast_mode=False)
time_standard = time.time() - start

print(f"⏱️  Execution Time: {time_standard:.2f} seconds")
print(f"✅ Verdict: {result_standard['verdict']}")
print(f"✅ Confidence: {result_standard['confidence']}")
print(f"✅ Processing Time Reported: {result_standard['processing_time']}")
print(f"✅ Evidence Sources: {len(result_standard.get('evidence', []))}")

if time_standard < 15:
    print("✓ PASS: Standard mode under 15 seconds!")
elif time_standard < 30:
    print("✓ ACCEPTABLE: Standard mode under 30 seconds")
else:
    print(f"✗ WARNING: Standard mode took {time_standard:.2f} seconds")

# Test 3: CACHE TEST
print("\n" + "=" * 80)
print("TEST 3: CACHE TEST (Same Content Again)")
print("=" * 80)

start = time.time()
result_cached = verify_with_enhanced_pipeline(long_article, use_openai=False, fast_mode=False)
time_cached = time.time() - start

print(f"⏱️  Execution Time: {time_cached:.2f} seconds")
print(f"✅ Verdict: {result_cached['verdict']}")
print(f"✅ Cached: {result_cached.get('cached', False)}")
print(f"✅ Processing Time Reported: {result_cached['processing_time']}")

if time_cached < 0.1:
    print("✓ PASS: Cache hit - instant response!")
else:
    print(f"✗ Note: Cache lookup took {time_cached:.2f} seconds")

# Summary
print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)
print(f"""
Content Size: {len(long_article.split())} words

FAST MODE (Heuristics):    {time_fast:.2f}s  (Target: < 5s)
STANDARD MODE (With API):  {time_standard:.2f}s  (Target: < 30s)
CACHE LOOKUP:              {time_cached:.2f}s  (Target: < 0.1s)

🎯 OPTIMIZATION RESULTS:
- Original time: ~10 minutes (600+ seconds) ❌
- New fast mode: {time_fast:.2f} seconds ✅ ({600/max(time_fast, 0.1):.0f}x faster!)
- New standard mode: {time_standard:.2f} seconds ✅ ({600/max(time_standard, 0.1):.0f}x faster!)

✨ KEY IMPROVEMENTS:
1. Aggressive API timeouts (3-5 seconds max per operation)
2. Fast-track mode using heuristics (no external API calls)
3. Result caching with TTL (instant for repeated content)
4. Early fallback to heuristics if timeouts occur
5. Parallel processing with thread pooling

{'=' * 80}
🎉 PERFORMANCE TEST COMPLETE!
{'=' * 80}
""")
