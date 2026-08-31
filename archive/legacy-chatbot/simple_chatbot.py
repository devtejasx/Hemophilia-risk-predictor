"""
Simple Standalone Chatbot - Works without API keys
Provides immediate responses to user questions
"""

import os
import json
from typing import Optional, List, Dict
from datetime import datetime

# Try to import OpenAI, but it's optional
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Knowledge base for hemophilia and general medical questions
KNOWLEDGE_BASE = {
    "inhibitor": {
        "keywords": ["inhibitor", "antibody", "immune", "factor viii", "factor ix"],
        "response": """
**About Inhibitors in Hemophilia:**
Inhibitors are antibodies your immune system develops against clotting factor replacement therapy. Here's what you should know:

• **What are they?** Antibodies that attack the factor replacement, making it less effective
• **Why they develop?** Immune system recognizes the replacement factor as foreign
• **Risk factors:** Early treatment, intensive use, genetic factors, some mutations have higher risk
• **Symptoms:** Bleeding that doesn't respond to usual factor doses, unexplained bruising
• **Detection:** Special blood tests measure inhibitor levels (Bethesda units)
• **Treatment:** High-dose factor, Immune Tolerance Induction (ITI), or bypassing agents

**Action Items:**
✅ Get regular inhibitor screening
✅ Monitor for unusual bleeding patterns
✅ Work with your hemophilia team if inhibitors develop
✅ Never stop treatment without consulting your doctor
"""
    },
    "mutation": {
        "keywords": ["mutation", "genetic", "gene", "factor", "dna", "hereditary"],
        "response": """
**About Hemophilia Mutations:**
Hemophilia is caused by mutations in genes that code for clotting factors. Here's the breakdown:

• **Factor VIII Gene** (Hemophilia A - 80% of cases)
  - Located on X chromosome
  - Over 2400 known mutations
  - Can be deletions, duplications, inversions, or point mutations

• **Factor IX Gene** (Hemophilia B - 20% of cases)
  - Also on X chromosome
  - Over 1000 known mutations

• **Inheritance Pattern:**
  - Males typically affected (have one X chromosome)
  - Females are usually carriers (have two X chromosomes)
  - ~30% of cases are new mutations

• **Severity Correlation:**
  - Large deletions = usually severe
  - Small mutations = variable severity
  - Same mutation can have different severity in different people

**Talk to your hemophilia team** about your specific mutation and what it means for your treatment!
"""
    },
    "treatment": {
        "keywords": ["treatment", "factor", "dose", "injection", "prophylaxis", "on-demand"],
        "response": """
**Hemophilia Treatment Options:**

**1. Replacement Therapy (Most Common)**
- **Factor Concentrates:** Derived from donated blood or genetically engineered
- **Prophylaxis:** Regular injections to prevent bleeding (2-3 times/week)
- **On-Demand:** Injection when bleeding starts

**2. Dosing Considerations:**
- Based on weight, age, severity, and activity level
- Adults typically: 25-50 units/kg for on-demand, 20-40 units/kg for prophylaxis
- More frequent dosing needed with high physical activity

**3. Product Types:**
- Standard half-life factors (8-12 hours)
- Extended half-life factors (up to 19 hours) - less frequent injections
- Bypassing agents for inhibitor patients

**4. New Therapies:**
- Gene therapy (permanent factor production)
- Non-factor therapies
- Long-acting subcutaneous treatments

**Important:**
- Never skip or change dose without doctor approval
- Track your doses and bleeding episodes
- Communicate with your hemophilia team about your lifestyle
- Good adherence = better prevention + fewer complications

**Ask your doctor** about what treatment is best for YOU!
"""
    },
    "exercise": {
        "keywords": ["exercise", "sport", "activity", "physical", "play", "run", "jump", "bleeding"],
        "response": """
**Physical Activity & Exercise with Hemophilia:**

**Safe Activities:**
✅ Swimming & water aerobics (low impact, great for joints)
✅ Cycling (stationary or outdoor with helmet/pads)
✅ Yoga & stretching (flexibility, joint health)
✅ Walking & hiking
✅ Golf, bowling
✅ Table tennis, badminton
✅ Strength training (controlled, supervised)

**Moderate Risk Activities:**
⚠️ Basketball, soccer (contact risk)
⚠️ Wrestling (direct contact)
⚠️ Skiing/snowboarding (high speed, trauma risk)
⚠️ Rock climbing

**High-Risk Activities:**
❌ American football, rugby (full contact)
❌ Boxing, martial arts
❌ Skateboarding tricks
❌ High-impact sports

**Key Rules:**
🛡️ Always wear protective gear (helmet, pads, joint supports)
💪 Build strength to support joints
🧘 Stretch regularly for flexibility
🩸 Take factor BEFORE playing sports
🤝 Tell your coach/trainer about hemophilia
🏥 Stop immediately if you feel unusual pain

**Remember:** Exercise is GOOD for you! Strong muscles protect your joints. Work with your hemophilia team to plan your activities.
"""
    },
    "monitoring": {
        "keywords": ["monitoring", "screening", "test", "checkup", "follow-up", "hospital", "clinic"],
        "response": """
**Regular Monitoring in Hemophilia:**

**Routine Tests:**
• Blood tests (factor levels, inhibitor screening, liver/kidney)
• Joint imaging (ultrasound or MRI for damage assessment)
• Viral screening (HIV, Hepatitis for blood product recipients)

**Recommended Schedule:**
- **Mild/Moderate:** Every 6-12 months
- **Severe:** Every 3-6 months minimum
- **After bleeding episodes:** ASAP or next available
- **New to therapy:** More frequent initially

**What Gets Checked:**
✅ Factor level (should be appropriate for your treatment)
✅ Inhibitors developed? (Yes/No, and strength if present)
✅ Liver function (if received blood products pre-screening)
✅ Viral antibodies (HIV, Hepatitis C, B)
✅ Joint structure (bleeding damage?)
✅ Overall bleeding control

**Between Visits:**
📝 Keep a bleeding diary
📋 Record all factor doses
📱 Report unusual symptoms immediately
🩸 Watch for: Unusual bruising, swelling, pain, nosebleeds, GI bleeding

**Important:** Regular monitoring helps catch problems early and adjust your treatment plan. Don't skip appointments!
"""
    },
    "risk": {
        "keywords": ["risk", "inhibitor risk", "dangerous", "complications", "safe", "danger"],
        "response": """
**Understanding Hemophilia Risks:**

**Short-term Risks:**
🩸 **Bleeding:** Can happen internally (joints, muscles) or externally
💔 **Intracranial Hemorrhage:** Brain bleeding (rare but serious)
🫀 **GI Bleeding:** Stomach/intestinal bleeding
🦵 **Joint Bleeds:** Repeated bleeding damages joints permanently

**Long-term Risks:**
🦵 **Joint Damage (Arthropathy):** Chronic pain, reduced mobility
🏥 **Hemophilic Arthropathy:** Irreversible joint destruction
🦠 **Infection:** From contaminated blood products (less common now)
🧬 **Inhibitor Development:** 5-20% of severe hemophilia A patients
💊 **Medication Side Effects:** Rare but possible

**Risk Factors for Inhibitor Development:**
- Family history of inhibitors
- Specific gene mutations
- Early/intensive factor exposure
- Frequent treatment
- Ethnic background (some are higher risk)
- Non-adherence to treatment

**How to Reduce Risk:**
✅ Take your prescribed prophylaxis
✅ Stay physically active (strengthens joints)
✅ Maintain healthy weight
✅ Regular medical monitoring
✅ Report symptoms immediately
✅ Work closely with hemophilia team
✅ Avoid unnecessary trauma

**Good news:** With modern treatment and management, most people with hemophilia live normal lifespans with good quality of life!
"""
    },
    "bleeding": {
        "keywords": ["bleeding", "bleed", "blood", "hemorrhage", "bleeding episode", "blood loss"],
        "response": """
**Recognizing & Managing Bleeding Episodes:**

**Where Bleeding Can Happen:**
- 🦵 **Joints** (knees, ankles, elbows) - most common
- 💪 **Muscles** (legs, arms, back)
- 🧠 **Brain** (rare but serious - go to ER)
- 🫀 **Internal organs** (serious - seek care)
- 👃 **Nose** (common, usually minor)
- 🪥 **Mouth/Gums** (after dental work)
- 🦴 **Bones** (fractures can bleed heavily)

**Symptoms to Watch:**
⚠️ Sudden swelling in a joint
⚠️ Warmth & redness in joint
⚠️ Severe pain/stiffness
⚠️ Inability to move limb normally
⚠️ Large unexplained bruise
⚠️ Headache (with fever = emergency)
⚠️ Vomiting/abdominal pain
⚠️ Chest pain or shortness of breath

**What to Do:**
1. **Take your factor immediately** (or call hemophilia center)
2. **R.I.C.E. (if joint bleeding):**
   - **R**est - stop activity
   - **I**ce - apply ice pack (15-20 min)
   - **C**ompress - wrap with elastic bandage
   - **E**levate - raise limb above heart
3. **Rest the bleeding area** for 24-48 hours
4. **Repeat factor dose** if needed (per your plan)
5. **See doctor if:** No improvement in 48 hours, very severe, head/chest/abdominal
6. **ER immediately:** Head injury, severe pain, hemoptysis, severe GI bleed

**Prevention:**
✅ Take prophylaxis as prescribed
✅ Wear protective equipment during activities
✅ Strengthen muscles (good support)
✅ Avoid trauma and risky behavior
✅ Stay aware of surroundings
"""
    },
    "diet": {
        "keywords": ["diet", "food", "nutrition", "vitamin", "eat", "healthy"],
        "response": """
**Nutrition for Hemophilia:**

**Foods That Support Clotting:**
✅ **Vitamin K** (important for factor synthesis):
   - Leafy greens: spinach, kale, broccoli, lettuce
   - Cabbage, Brussels sprouts
   - Green tea

✅ **Protein** (for factor production):
   - Lean meats, fish, chicken
   - Eggs, dairy
   - Legumes (beans, lentils)
   - Nuts & seeds

✅ **Iron** (for blood health):
   - Red meat, poultry
   - Fish & shellfish
   - Fortified cereals
   - Dried fruit

✅ **Antioxidants** (joint health):
   - Berries (blueberries, strawberries)
   - Citrus fruits
   - Nuts
   - Olive oil

❌ **Foods to Limit:**
- High fat (can affect digestion)
- Excessive alcohol (thins blood)
- Hot spicy foods (GI irritation)
- Hard/crunchy (injury to mouth/throat)

**General Tips:**
🥗 Eat balanced meals 3x daily
💧 Stay hydrated (water, electrolytes)
⚖️ Maintain healthy weight (less joint stress)
🍎 Eat fresh, whole foods when possible
🏋️ Combine with exercise for best results

**Important:** If you have food restrictions, ask your doctor or nutritionist!
"""
    },
    "pregnancy": {
        "keywords": ["pregnancy", "pregnant", "baby", "family", "children", "birth", "female"],
        "response": """
**Hemophilia & Family Planning:**

**For Women Carriers:**
👶 **Can you have children?** YES! Pregnancy is possible and usually safe.

• **Your children's risk:**
  - 50% sons: hemophilia A or B
  - 50% daughters: carriers (rarely affected)

• **During pregnancy:**
  - Factor VIII naturally rises (good news for A!)
  - Factor IX doesn't change (B more risky)
  - Regular monitoring recommended
  - Delivery options: vaginal or C-section possible
  - Plan labor at hemophilia center if possible

**For Men with Hemophilia:**
👶 **Can you have children?** YES! Your condition doesn't prevent fatherhood.

• **Your children:**
  - All daughters will be carriers (get one X from you)
  - Sons will NOT inherit hemophilia (get Y from you)
  - Daughters won't have hemophilia (protected by mother's X)

• **Genetic counseling recommended** for informed decisions

**Pre-Conception Planning:**
✅ See hemophilia team (establish baseline factor levels)
✅ Genetic counseling (understand inheritance)
✅ Optimize disease control
✅ Discuss medication safety during pregnancy
✅ Plan delivery at specialized center

**Prenatal Testing:**
- Genetic testing available (amniocentesis, CVS)
- Helps plan for baby's needs
- Discuss with OB/GYN and hemophilia team

**Support:**
- Many people with hemophilia have healthy children
- Modern treatment makes this safer
- Speak with counselors who've been through it
"""
    },
}

# General response templates for questions not in knowledge base
GENERAL_RESPONSES = {
    "greeting": "Hello! I'm your medical AI assistant. I'm here to help answer questions about hemophilia, health, and general wellness. What would you like to know?",
    
    "help": """
I can help you with questions about:
• Hemophilia (Type A & B)
• Inhibitors and antibodies
• Treatment options
• Exercise and physical activity
• Diet and nutrition
• Pregnancy and family planning
• Monitoring and checkups
• Joint health and complications
• Or ANY other health/general knowledge questions!

Just ask me anything and I'll do my best to help!
""",
    
    "unknown": "I'm not sure about that specific topic. However, I'd recommend discussing it with your hemophilia team or doctor. Is there anything else about hemophilia or health I can help you with?",
}


def find_knowledge_match(user_message: str) -> Optional[str]:
    """Find a response from knowledge base for user message"""
    user_msg_lower = user_message.lower()
    
    for topic, info in KNOWLEDGE_BASE.items():
        for keyword in info["keywords"]:
            if keyword in user_msg_lower:
                return info["response"]
    
    return None


def try_openai_response(user_message: str, api_key: Optional[str] = None) -> Optional[str]:
    """Try to get response from OpenAI API"""
    if not HAS_OPENAI:
        return None
    
    try:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful medical AI assistant. Answer questions clearly and helpfully. Always recommend consulting a doctor for serious medical concerns."},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=500,
            timeout=10
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"OpenAI error: {e}")
        return None


def get_chatbot_response(user_message: str, api_key: Optional[str] = None) -> str:
    """
    Get chatbot response with fallback strategy:
    1. Try knowledge base match
    2. Try OpenAI API
    3. Return generic response
    """
    
    user_msg_lower = user_message.lower().strip()
    
    # Check for help/greeting
    if any(word in user_msg_lower for word in ["help", "what can", "assistance"]):
        return GENERAL_RESPONSES["help"]
    
    if any(word in user_msg_lower for word in ["hello", "hi", "hey", "greetings"]):
        return GENERAL_RESPONSES["greeting"]
    
    # Try knowledge base first
    kb_response = find_knowledge_match(user_message)
    if kb_response:
        return kb_response
    
    # Try OpenAI API
    api_response = try_openai_response(user_message, api_key)
    if api_response:
        return api_response
    
    # Fallback
    return GENERAL_RESPONSES["unknown"]


def save_conversation(messages: List[Dict], patient_name: str = "Anonymous") -> str:
    """Save conversation to file for record keeping"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{patient_name}_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"Chat History - {patient_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            for msg in messages:
                role = "You" if msg["role"] == "user" else "Assistant"
                f.write(f"{role}:\n{msg['content']}\n\n")
        
        return filename
    except Exception as e:
        print(f"Error saving conversation: {e}")
        return None
