# 🤖 GPT Integration Setup Guide

Your Fake News Detector now supports **ChatGPT-like conversations** using OpenAI's API! This makes the AI much more natural and helpful for general questions.

## 🚀 Quick Setup

### 1. **Get OpenAI API Key**
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up or log in to your account
3. Click "Create new secret key"
4. Copy your API key (starts with `sk-proj-...`)

### 2. **Configure Your Project**
1. Copy `env_example.txt` to `.env` in your project folder:
   ```bash
   cp env_example.txt .env
   ```

2. Edit `.env` file and add your API key:
   ```
   OPENAI_API_KEY=sk-proj-your-actual-api-key-here
   ```

### 3. **Install Required Package**
```bash
pip install openai>=0.28.0 python-dotenv
```

### 4. **Restart Django Server**
```bash
python manage.py runserver
```

## ✨ What You Get

### **🧠 Enhanced Conversations**
- **Natural responses** like ChatGPT for general questions
- **Context awareness** - remembers conversation flow
- **Intelligent routing** - automatically detects when to chat vs fact-check
- **Fallback system** - works even without API key (basic responses)

### **🎯 Smart Detection**
The AI now intelligently handles:
- **General Questions**: "What is artificial intelligence?" → GPT response
- **Casual Chat**: "How are you?" → Natural conversation
- **Fact-Checking**: "Is this news article true?" → Analysis mode
- **URLs**: "Check this link" → Fact-checking mode

### **💬 Example Conversations**

**Before (Basic):**
```
User: What do you think about climate change?
AI: I'm here to help with fact-checking. Share content to verify.
```

**After (GPT-Powered):**
```
User: What do you think about climate change?
AI: Climate change is a significant global challenge backed by overwhelming scientific consensus. The evidence shows human activities, particularly greenhouse gas emissions, are driving unprecedented changes in Earth's climate system. I can help fact-check specific claims about climate science if you'd like to share any articles or statements!
```

## 🔧 Technical Details

### **How It Works**
1. **Intent Detection**: AI determines if input is conversational or needs fact-checking
2. **GPT Integration**: Conversational queries go to OpenAI's GPT-3.5-turbo
3. **Fact-Checking**: News/claims go to the specialized detection models
4. **Fallback**: If GPT fails, uses enhanced local responses

### **API Usage & Costs**
- Uses **GPT-3.5-turbo** (most cost-effective)
- Typical cost: **~$0.002 per conversation**
- Includes conversation context (last 10 messages)
- Responses limited to 500 tokens for efficiency

### **Privacy & Security**
- API key stored securely in environment variables
- Conversations sent to OpenAI (review their privacy policy)
- Local fallback available if you prefer not to use external APIs

## 🛠️ Troubleshooting

### **"OpenAI package not installed"**
```bash
pip install openai>=0.28.0
```

### **"API key not found"**
- Check your `.env` file exists and has the correct key
- Restart Django server after adding the key
- Verify key starts with `sk-proj-` or `sk-`

### **"GPT API Error"**
- Check your OpenAI account has credits
- Verify API key is valid and active
- System will automatically fall back to local responses

### **Still Using Basic Responses?**
- Ensure `.env` file is in the correct directory
- Check Django logs for initialization messages
- Verify `python-dotenv` is installed

## 🎉 Success Indicators

When working correctly, you'll see:
- **Console**: "✅ OpenAI API initialized successfully"
- **Responses**: More natural, detailed answers to questions
- **API indicator**: Responses show "AI Assistant" powered by GPT

## 💡 Tips

1. **Test with questions**: Try "What is machine learning?" or "How does AI work?"
2. **Mix conversations**: Ask questions, then share news to fact-check
3. **Check costs**: Monitor your OpenAI usage dashboard
4. **Fallback ready**: System works great even without API key

---

**Your AI is now much smarter and more conversational! 🎉**

For support, check the Django console logs or create an issue in the repository.
