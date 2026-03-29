"""
Local Pre-trained GPT Model for Chatbot
Provides offline question-answering capability without API dependency
"""

import os
from typing import Optional, List, Dict

# Try to use transformers for local inference
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Fallback knowledge base for when transformers is not available
KNOWLEDGE_BASE = {
    "medical": {
        "what is hemophilia": """
Hemophilia is a rare bleeding disorder where blood doesn't clot properly. There are two main types:
- Hemophilia A: Missing clotting factor VIII
- Hemophilia B: Missing clotting factor IX

Treatment involves regular factor replacement therapy to help your blood clot normally.
""",
        "treatment": """
Hemophilia is treated with clotting factor replacement therapy:
- Prophylaxis: Regular preventive dosing
- On-demand: Treatment when bleeding occurs
- Extended half-life products: Last longer in your system
- Novel therapies: New bispecific antibodies available

Work with your hemophilia team to choose the best option for you.
""",
        "risk": """
Inhibitor risk varies by person based on:
- Type and severity of hemophilia
- Genetics and family history
- Treatment exposure
- Adherence to therapy

Regular monitoring helps catch problems early. Your team will assess your specific risk.
""",
    },
    "lifestyle": {
        "exercise": """
Safe activities for hemophilia:
- Walking and jogging
- Swimming
- Cycling with proper gear
- Low-impact aerobics
- Yoga

Avoid contact sports without doctor approval. Warm up before exercise and keep factor on hand.
""",
        "diet": """
Good nutrition for hemophilia:
- Protein: Chicken, fish, eggs
- Iron: Red meat, spinach, beans
- Calcium: Dairy, fortified foods
- Fruits & vegetables: Full of vitamins
- Whole grains: For fiber

Avoid excessive alcohol as it increases bleeding risk.
""",
        "stress": """
Managing stress helps your health:
- Deep breathing exercises
- Regular exercise
- Meditation or mindfulness
- Talk to friends/family
- Hobbies and relaxation
- Professional counseling if needed

Stress management improves treatment adherence and overall well-being.
""",
    },
    "general": {
        "hello": "Hello! I'm your AI health assistant. How can I help you today?",
        "hi": "Hi there! I'm here to answer questions about hemophilia or general health. What would you like to know?",
        "thanks": "You're welcome! Feel free to ask me anything else.",
        "help": "I can help with questions about hemophilia, treatment, lifestyle, health, and general knowledge. What would you like to know?",
    }
}


class LocalChatbot:
    """Local pre-trained GPT model for offline chatbot"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.use_local_model = False
        
        # Try to load pre-trained model
        if HAS_TRANSFORMERS:
            try:
                # Use a lightweight model that's fast and good for Q&A
                self.load_model()
                self.use_local_model = True
            except Exception as e:
                print(f"Could not load local model: {e}. Using fallback...")
                self.use_local_model = False
    
    def load_model(self):
        """Load pre-trained GPT model"""
        try:
            # Using DistilGPT-2 - lightweight and efficient
            model_name = "distilgpt2"
            print(f"Loading {model_name}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            print("✅ Local model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            self.model = None
            self.tokenizer = None
    
    def generate_response(self, question: str, context: Optional[Dict] = None, max_length: int = 150) -> str:
        """Generate response using local model or fallback"""
        
        # First check knowledge base for exact matches
        response = self._check_knowledge_base(question)
        if response:
            return response
        
        # If local model is available, use it
        if self.use_local_model and self.model and self.tokenizer:
            try:
                return self._generate_with_model(question, max_length)
            except Exception as e:
                print(f"Model generation failed: {e}")
                return self._fallback_response(question, context)
        
        # Otherwise use fallback
        return self._fallback_response(question, context)
    
    def _check_knowledge_base(self, question: str) -> Optional[str]:
        """Check if question matches knowledge base"""
        q_lower = question.lower()
        
        # Check all categories
        for category, qa_pairs in KNOWLEDGE_BASE.items():
            for key, value in qa_pairs.items():
                if key in q_lower or any(keyword in q_lower for keyword in key.split()):
                    return value
        
        return None
    
    def _generate_with_model(self, question: str, max_length: int) -> str:
        """Generate response using pre-trained model"""
        try:
            # Prepare input
            input_text = f"Q: {question}\nA:"
            
            # Tokenize
            inputs = self.tokenizer.encode(input_text, return_tensors="pt")
            
            # Generate
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response
            if "A:" in response:
                response = response.split("A:")[1].strip()
            
            return response if response else self._fallback_response(question)
        
        except Exception as e:
            print(f"Generation error: {e}")
            return self._fallback_response(question)
    
    def _fallback_response(self, question: str, context: Optional[Dict] = None) -> str:
        """Generate intelligent fallback response"""
        
        q_lower = question.lower()
        
        # Recognize question type
        if any(word in q_lower for word in ["how", "what", "why", "when", "where", "who"]):
            question_type = "question"
        elif any(word in q_lower for word in ["can i", "should i", "is it okay"]):
            question_type = "permission"
        elif any(word in q_lower for word in ["tell me", "explain", "describe"]):
            question_type = "request"
        else:
            question_type = "general"
        
        # Build response based on question type
        if question_type == "permission":
            return f"""
Based on your question about "{question}": 
This depends on your specific hemophilia severity and treatment plan. 

**I recommend:**
1. Ask your hemophilia treatment team first
2. Ensure you have factor therapy available
3. Start slowly and monitor how you feel
4. Keep emergency contacts ready

Your medical team knows your complete profile and can give personalized advice.
"""
        elif question_type == "request":
            return f"""
**About: {question}**

Here's what I know:
- Every person with hemophilia is different
- Your treatment team can provide detailed guidance
- The system learns from similar patients to help recommendations
- Your safety is the priority

**Next steps:**
1. Discuss with your hemophilia team
2. Let me know if you have specific concerns
3. I can provide general information to help your discussion

What specific aspect would you like to know more about?
"""
        else:
            # General question
            if context and "Name" in context:
                patient_name = context.get("Name", "")
                severity = context.get("Severity", "")
                return f"""
Your question: "{question}"

{f"Based on your profile ({patient_name}, {severity} hemophilia):" if severity else ""}
I can help by providing general information about:
- Your specific hemophilia type and severity
- Treatment options and monitoring
- Lifestyle and activities
- Managing your condition

**For personalized medical advice**, please discuss with your hemophilia treatment team.

Would you like to know more about any specific topic?
"""
            else:
                return f"""
Great question: "{question}"

I'm here to help with information about hemophilia and health. While I aim to be helpful, medical decisions should be made with your treatment team.

**I can discuss:**
- Hemophilia types and severity
- Treatment strategies
- Lifestyle modifications
- General health and wellness
- Monitoring and prevention

What would you like to know more about?
"""


def get_local_response(question: str, patient_context: Optional[Dict] = None) -> str:
    """
    Get response from local chatbot
    This is the main function to use for local inference
    """
    try:
        # Initialize chatbot (singleton pattern could be used in production)
        chatbot = LocalChatbot()
        response = chatbot.generate_response(question, patient_context)
        return response
    except Exception as e:
        # Ultra-safe fallback
        return f"""
Thanks for your question: "{question}"

I'm processing your request. If you have a specific concern about hemophilia treatment or health, I recommend:
1. Contacting your hemophilia treatment team
2. Reaching out to patient organizations
3. Consulting with your healthcare provider

I'm here to provide general information to support those conversations.
"""


# For testing
if __name__ == "__main__":
    print("🤖 Local Chatbot Test\n")
    
    chatbot = LocalChatbot()
    
    test_questions = [
        "What is hemophilia?",
        "Can I exercise?",
        "What should I eat?",
        "Hello there",
        "Tell me about my risk",
        "Is it safe to travel with hemophilia?",
    ]
    
    for question in test_questions:
        print(f"\n❓ Q: {question}")
        response = chatbot.generate_response(question)
        print(f"✅ A: {response[:200]}...")
