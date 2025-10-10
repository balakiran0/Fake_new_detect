#!/usr/bin/env python3
"""
Test script for Enhanced Fake News Verification System
Demonstrates the 8-step verification pipeline
"""

import os
import sys
import django
import json
from datetime import datetime

# Setup Django environment
sys.path.append(r'c:\Users\Bala Kiran\Desktop\fake_news_detector\fakenews_project')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fakenews_project.settings')
django.setup()

from detector.enhanced_verification import verify_with_enhanced_pipeline

def test_enhanced_verification():
    """Test the enhanced verification system with sample claims"""
    
    print("🔍 Enhanced Fake News Verification System Test")
    print("=" * 60)
    
    # Test cases from your example
    test_cases = [
        {
            "name": "NASA Earth Blackout Claim",
            "text": "NASA announced Earth will go dark for 3 days next month due to solar alignment."
        },
        {
            "name": "COVID-19 Vaccine Claim", 
            "text": "Breaking: New study shows COVID-19 vaccines contain microchips for tracking, according to researchers at Johns Hopkins University."
        },
        {
            "name": "Climate Change Claim",
            "text": "Scientists confirm that global warming is a hoax created by the government to control people, new leaked documents reveal."
        },
        {
            "name": "Election Fraud Claim",
            "text": "Exclusive: Voting machines were hacked in 2020 election, changing millions of votes according to cybersecurity experts."
        },
        {
            "name": "Health Misinformation",
            "text": "Drinking bleach can cure cancer and COVID-19, claims viral social media post shared by thousands."
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['name']}")
        print("-" * 50)
        print(f"📝 Claim: {test_case['text']}")
        print("\n⏳ Running enhanced verification...")
        
        try:
            # Run verification
            result = verify_with_enhanced_pipeline(test_case['text'], use_openai=True)
            
            # Display results
            print(f"\n✅ VERIFICATION RESULTS:")
            print(f"🎯 Verdict: {result['verdict']}")
            print(f"📊 Confidence: {result['confidence']}")
            print(f"🔍 Core Claim: {result.get('core_claim', 'N/A')}")
            print(f"📖 Explanation: {result.get('explanation', 'N/A')}")
            
            # Show entities if available
            entities = result.get('entities', {})
            if entities:
                print(f"\n🏷️ Extracted Entities:")
                for entity_type, entity_list in entities.items():
                    if entity_list:
                        print(f"  • {entity_type.title()}: {', '.join(entity_list)}")
            
            # Show evidence sources
            evidence = result.get('evidence', [])
            if evidence:
                print(f"\n📚 Evidence Sources ({len(evidence)} found):")
                for j, source in enumerate(evidence[:3], 1):  # Show top 3
                    print(f"  {j}. {source['source']}: {source['title']}")
                    print(f"     🔗 {source['url']}")
                    print(f"     ⚖️ Verdict: {source.get('verdict', 'N/A')}")
            
            # Show bias analysis
            bias_analysis = result.get('bias_analysis', {})
            if bias_analysis:
                print(f"\n🎭 Bias Analysis:")
                print(f"  • Source Diversity: {bias_analysis.get('source_diversity', 0)}")
                print(f"  • Fact-Check Coverage: {bias_analysis.get('fact_check_coverage', 0)}")
                print(f"  • Reliability Score: {bias_analysis.get('reliability_score', 0):.2f}")
                
                suspicious_patterns = bias_analysis.get('suspicious_patterns', [])
                if suspicious_patterns:
                    print(f"  ⚠️ Suspicious Patterns: {', '.join(suspicious_patterns)}")
            
            print(f"\n⏱️ Processing Time: {result.get('processing_time', 'N/A')}")
            print(f"💾 Cached: {'Yes' if result.get('cached', False) else 'No'}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("\n" + "=" * 60)

def test_api_integration():
    """Test API integration"""
    print("\n🌐 API Integration Test")
    print("=" * 40)
    
    try:
        import requests
        
        # Test the enhanced verification API endpoint
        url = "http://localhost:8000/api/enhanced-verify/"
        test_data = {
            "text": "NASA announced Earth will go dark for 3 days next month."
        }
        
        print(f"📡 Testing API endpoint: {url}")
        print(f"📤 Sending: {test_data['text']}")
        
        response = requests.post(url, json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API Response: {result['verdict']} ({result['confidence']})")
            print(f"⏱️ Total Processing Time: {result.get('total_processing_time', 'N/A')}")
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            
    except ImportError:
        print("⚠️ Requests library not available for API testing")
    except Exception as e:
        print(f"❌ API Test Error: {str(e)}")

if __name__ == "__main__":
    print(f"🚀 Starting Enhanced Verification Tests at {datetime.now()}")
    
    # Test the verification system
    test_enhanced_verification()
    
    # Test API integration (optional)
    print("\n" + "=" * 60)
    test_api_integration()
    
    print(f"\n🏁 Tests completed at {datetime.now()}")
    print("\n💡 Usage Instructions:")
    print("1. For regular analysis: Use /api/detect/ (automatically chooses method)")
    print("2. For enhanced verification: Use /api/enhanced-verify/ (always uses 8-step pipeline)")
    print("3. Enhanced verification is automatically used for:")
    print("   • Content longer than 50 words")
    print("   • Content detected as 'news article'")
    print("   • Content with URLs or formal claims")
