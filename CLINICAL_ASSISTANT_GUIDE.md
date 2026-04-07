# Structured Clinical AI Assistant - Complete Guide

## Overview

The **Structured Clinical AI Assistant** is an advanced clinical decision support system built into the hemophilia AI platform. It provides mode-based clinical reasoning with patient context integration, safety disclaimers, and evidence-based prompt templates.

## Key Features

### ✅ 4 Specialized Clinical Modes

1. **🔍 Diagnosis Support**
   - Interpret symptoms and clinical findings
   - Support differential diagnosis reasoning
   - Analyze mutation implications
   - Consider hemophilia-specific presentations

2. **💊 Treatment Recommendation**  
   - Optimize treatment dosing strategies
   - Evaluate treatment modality options
   - Address treatment adherence challenges
   - Compare treatment efficiency

3. **⚠️ Risk Explanation**
   - Interpret risk assessment scores
   - Identify protective factors
   - Explain inhibitor development risks
   - Contextualize individual risk profiles

4. **📊 Monitoring Analysis**
   - Guide appropriate monitoring frequency
   - Design monitoring protocols
   - Identify monitoring milestones
   - Interpret monitoring results

### ✨ Core Capabilities

✅ **Patient Context Integration** - Automatically includes patient data:
- Demographics: Name, Age, Gender, Ethnicity
- Clinical: Severity, Mutation, Factor Level
- Treatment: Dose, Product, Exposure
- Risk: Inhibitor status, Risk score
- History: Bleeding episodes, Joint damage

✅ **Safety Disclaimers** - Prominent warnings on all responses:
- Clear statement: "AI suggestions are NOT medical advice"
- Emphasis on specialist consultation requirement
- Educational information clarification
- Never replaces professional judgment

✅ **Multi-turn Conversation** - Context-aware dialogue:
- Maintains conversation history per mode
- Tracks conversation mode and patient context
- Supports follow-up questions
- Preserves clinical reasoning chain

✅ **Mode-Specific Templates** - Each mode has:
- Structured system prompt with guidelines
- Patient context formatting rules
- Clinical reasoning framework
- Safety disclaimer integration

✅ **Medical Terminology Support** - 18 hemophilia-specific terms:
- Mutation definitions
- Product types
- Clinical conditions
- Management strategies

## Architecture

### Module Structure: `clinical_assistant.py`

```python
# Main Classes
ClinicalAssistantMode        # Enum-like mode constants
StructuredPromptTemplates    # Static prompt template methods
StructuredClinicalAssistant  # Main orchestration class

# Helper Functions
get_clinical_response()      # Main user-facing function
get_available_modes()        # Mode metadata retrieval
```

### Integration Point: `app.py` Line 2082+

The Advanced Chatbot page uses:
- Mode selector (4 buttons for mode selection)
- Patient context display (if available)
- Chat interface with medical styling
- Example questions per mode
- Conversation history with export

## User Interface

### Layout

**Page Structure:**
```
┌─ 🤖 Structured Clinical AI Assistant
│  ├─ ⚠️ CRITICAL MEDICAL DISCLAIMER (prominent warning box)
│  ├─ 📋 Patient Context (if available: Name, Mutation, Severity, Risk, Dose)
│  ├─ 🎯 Mode Selection (4 clickable buttons)
│  │  ├─ 🔍 Diagnosis Support
│  │  ├─ 💊 Treatment Recommendation
│  │  ├─ ⚠️ Risk Explanation
│  │  └─ 📊 Monitoring Analysis
│  ├─ 💬 Chat Tab
│  │  ├─ Conversation display with avatars
│  │  ├─ AI response disclaimer ("AI-generated response for educational discussion")
│  │  ├─ Response action buttons (👍 Helpful, 📋 Copy, ⚕️ Review)
│  │  ├─ Chat input area
│  │  └─ Send button
│  ├─ 📚 Examples Tab (4 mode-specific example questions)
│  └─ 📝 History Tab (conversation export & management)
└─
```

### Mode Buttons

Each mode button shows:
- 🔍 icon
- Mode name (abbreviated)
- Click state indicator (✓ when selected)
- Tooltip with full description

Clicking a mode:
- Switches to that mode
- Clears conversation history (for clean slate)
- Updates example questions
- Refreshes the UI

### Chat Interface

**Features:**
- User messages: Avatar (👤), regular text
- AI responses: Avatar (⚕️), disclaimer + content
- Feedback buttons: Helpful, Copy, Review
- Input area: Large textarea with placeholder text
- Send button: 🚀 Send with visual feedback

**Conversation Flow:**
1. User types question
2. System shows "Processing..." indicator
3. AI generates response using structured prompt
4. Response displays with disclaimer
5. Medical context shown if applicable

### Examples Tab

**4 Mode-Specific Examples Each:**

*Diagnosis Support:*
- "How do we interpret a high factor level with persistent bleeding?"
- "What's the differential diagnosis for joint swelling in my patient?"
- "How do we differentiate inhibitor-related bleeding from other causes?"

*Treatment Recommendation:*
- "Is my patient's current dose appropriate for their severity?"
- "What are the advantages of switching to extended half-life products?"
- "How do we optimize treatment adherence in this patient?"

*Risk Explanation:*
- "Why is my patient's inhibitor risk score elevated?"
- "What are the protective factors we should consider?"
- "How does family history impact this patient's risk?"

*Monitoring Analysis:*
- "What monitoring frequency is appropriate for this risk level?"
- "When should we perform inhibitor screening?"
- "What are the key indicators of treatment failure?"

Clicking an example:
- Sends it as a query
- Shows loading indicator
- Displays AI response
- Adds to conversation history

### History Tab

**Features:**
- Conversation summary display
- Export to text file with timestamp
- Clear history button
- Group by mode view
- Question preview list

**Export Format:**
```
[MODE_NAME]
You: Your question
AI: AI response
---

[MODE_NAME]
You: Next question
...
```

## Prompt Templates

Each mode template includes:

1. **System Context**
   - Role clarification: "You are a clinical decision support AI"
   - Hemophilia expertise emphasis
   - Patient context availability notification

2. **Patient Context Section**
   - Formatted patient data (if available)
   - Key clinical parameters highlighted
   - Risk factors summarized

3. **Mode-Specific Guidelines** (3-7 bullet points)
   - Focus area for this mode
   - Clinical reasoning approach
   - Key considerations
   - Evidence basis

4. **Clinical Reasoning Framework**
   - Structured thinking approach
   - Decision-making process
   - Consideration factors
   - Evidence integration

5. **Critical Disclaimer**
   ```
   CRITICAL: AI-generated suggestions are for [mode purpose] only 
   and should NOT replace professional medical judgment. 
   This is educational information to support clinical discussion, 
   not medical advice.
   ```

## Safety & Disclaimer Structure

### Disclaimer Locations

1. **Top of Page** (Prominent Warning Box)
   - Bold header: "⚠️ CRITICAL MEDICAL DISCLAIMER"
   - Clear statement of AI limitations
   - Required clinical actions
   - Emphasis that AI is a support tool

2. **Before Each Response**
   - Statement: "AI-generated response for educational discussion"
   - Reminds user of non-medical nature

3. **In Every Prompt Template**
   - Embedded in system prompt
   - Critical tone: "CRITICAL: AI-generated suggestions are for [purpose] only"
   - Clear non-advisory nature

### Required User Behaviors

✓ Always consult qualified hematologists
✓ Use AI suggestions to prepare for appointments
✓ Never delay medical treatment based on AI
✓ Report adverse events immediately

## Patient Data Integration

### Available Patient Context

From `st.session_state.data`:

```python
{
    'Name': string,              # Patient name
    'Age': int,                  # Age in years
    'Gender': string,            # Gender
    'Ethnicity': string,         # Ethnicity
    'Severity': string,          # Mild/Moderate/Severe
    'Mutation': string,          # Gene mutation name
    'Blood Type': string,        # ABO/Rh blood type
    'HLA Type': string,          # HLA type
    'Dose': float,               # Treatment dose in units
    'Exposure': int,             # Days of treatment exposure
    'Product': string,           # Product name/type
    'Adherence': float,          # Adherence percentage
    'Family History': string,    # Family medical history
    'Previous Inhibitor': bool,  # Inhibitor history
    'Joint Damage': int,         # Number of affected joints
    'Bleeding Episodes': int,    # Monthly average
    'Factor Level': float,       # Factor activity level %
    'Immunosuppression': bool,   # Immunosuppression status
    'Active Infection': bool,    # Active infection present
    'Vaccination': string,       # Vaccination status
    'Activity Level': string,    # High/Moderate/Low
    'Stress Level': string,      # High/Moderate/Low
    'Risk': float                # Risk score (0-1)
}
```

### Context Formatting

Patient context is formatted as readable summary:
```
Patient: [Name], [Age]yo [Gender]
Severity: [Severity]
Mutation: [Mutation]
Treatment: [Dose] units × [Product] × [Exposure] days
Risk Score: [Risk]%
Adherence: [Adherence]%
Key Issues: Joint damage ([#]), Bleeding episodes ([#]/month)
```

### No Patient Data Scenario

If no patient data available:
- General hemophilia mode available
- Examples use generic patient scenarios
- Prompts ask for contextual information
- Can be used for educational purposes

## Operation Flow

### Query Processing

```
1. User selects mode
2. User enters question
3. Click "Send"
4. Show "Processing..." indicator
5. Build prompt:
   - Mode-specific template
   - Patient context (if available)
   - Conversation history (last 5 messages)
   - Current question
6. Call OpenAI GPT-4 API
7. Stream response to UI
8. Add to conversation history
9. Support action buttons appear
```

### Error Handling

**Graceful Fallbacks:**

| Error Type | Handling |
|-----------|----------|
| No API Key | "API Connection Issue" message |
| Rate Limited (429) | "System busy, please retry" |
| Invalid Key (401) | "Authentication failed" |
| Network Error | Generic error with advice to retry |
| Timeout | "Taking longer than expected" |

## Configuration

### Mode Constants

```python
ClinicalAssistantMode.DIAGNOSIS_SUPPORT      # "diagnosis_support"
ClinicalAssistantMode.TREATMENT_RECOMMENDATION # "treatment_recommendation"
ClinicalAssistantMode.RISK_EXPLANATION        # "risk_explanation"
ClinicalAssistantMode.MONITORING_ANALYSIS     # "monitoring_analysis"
```

### Template Customization

To modify a prompt template, edit `clinical_assistant.py`:

```python
@staticmethod
def diagnosis_support(patient_context: str, question: str) -> str:
    return f"""
    [System prompt content]
    {patient_context}
    [Guidelines]
    Question: {question}
    """
```

### Adding Medical Terms

To add hemophilia terminology:

```python
MEDICAL_DEFINITIONS = {
    "new_term": "Definition here",
    ...
}
```

## Usage Examples

### Example 1: Patient with Inhibitor Risk

**Setup:**
- Patient: John, 12yo, Moderate severity
- Mutation: Intron 22 inversion
- Risk: 45%

**Mode:** Risk Explanation
**Question:** "Why is my risk score 45%?"

**Response Includes:**
- Patient context with all relevant data
- Risk factor breakdown
- Protective factors explanation
- Monitoring recommendations

### Example 2: Treatment Optimization

**Setup:**
- Patient: Maria, 28yo, Severe
- Current: 2000 IU 3x/week
- Adherence: 60%

**Mode:** Treatment Recommendation
**Question:** "Is my current dose appropriate?"

**Response Includes:**
- Patient context review
- Dose optimization analysis
- Adherence impact discussion
- Alternative regimen suggestions

### Example 3: Diagnosis Support

**Setup:**
- Patient: Ahmed, 35yo, Moderate
- Recent: Shoulder pain, swelling

**Mode:** Diagnosis Support
**Question:** "What could be causing the shoulder pain?"

**Response Includes:**
- Patient context with hemophilia history
- Differential diagnosis for hemophilia patient
- Joint damage risk consideration
- Investigation recommendations

### Example 4: Monitoring Protocol

**Setup:**
- Patient: Sofia, 8yo, Severe, High risk
- New inhibitor screening needed

**Mode:** Monitoring Analysis
**Question:** "What's the optimal inhibitor screening schedule?"

**Response Includes:**
- Patient context with risk level
- Evidence-based screening intervals
- Red flag symptoms
- Documentation requirements

## Troubleshooting

### Issue: No Response from AI

**Causes:**
- OpenAI API key missing or invalid
- Network connectivity issue
- API rate limit exceeded
- Timeout in API call

**Solutions:**
1. Verify .env file has `OPENAI_API_KEY`
2. Check internet connection
3. Wait 60 seconds and retry (rate limit)
4. Check API usage at openai.com

### Issue: Wrong Mode Selected

**Solution:**
- Click the correct mode button
- Conversation history clears automatically
- New mode-specific templates apply

### Issue: Patient Data Not Showing

**Causes:**
- Patient form not completed
- Session state cleared
- Data not saved properly

**Solutions:**
1. Complete Patient Form first
2. Ensure "Save & Process" button clicked
3. Check browser console for errors

### Issue: Medical Terms Not Clear

**Solution:**
- Medical terminology reference available in patient data
- Each term linked to evidence-based definitions
- Hover over terms for definitions (if implemented)

## Performance & Optimization

### Response Time

- Average: 2-5 seconds (depends on API)
- With slow network: 5-15 seconds
- Long questions: May take longer

### Token Usage

- Short question: ~500 tokens
- With patient context: ~800-1200 tokens
- Long conversation: ~2000+ tokens

### Cost Estimates

Per query (GPT-4):
- Simple: $0.02-0.05
- With context: $0.05-0.10
- Typical usage: $1-3 per session

## Compliance & Safety

### HIPAA Considerations

⚠️ **Important:**
- Do NOT share actual patient PHI through this system
- System not HIPAA-compliant as-is
- For production: Implement encryption and audit logging
- Consider anonymization of patient data

### Clinical Validity

- AI responses are for educational support only
- Not validated as clinical decision support tools
- Not FDA-approved or cleared
- Should not be used for diagnostic purposes
- Always require specialist review

### Documentation

- Maintain conversation logs for audit trail
- Document AI contribution to decisions
- Ensure human clinician review recorded
- Follow institutional policies

## Future Enhancements

Planned improvements:
- [ ] Real-time feedback integration
- [ ] Multi-language support
- [ ] Integration with external guidelines (ISTH, CDC)
- [ ] Explanation graphs (SHAP-style)
- [ ] Voice input/output
- [ ] Mobile app support
- [ ] Institutional customization
- [ ] Research data export

## Support & Documentation

**Main Files:**
- `clinical_assistant.py` - Core implementation (600+ lines)
- `app.py` lines 2082-2387 - UI integration
- `CLINICAL_ASSISTANT_GUIDE.md` - This file

**Related Documentation:**
- `README.md` - Platform overview
- `CONFIGURATION.md` - Setup instructions
- CHATBOT_GUIDE.md` - Legacy chatbot docs

**Contact & Issues:**
- Report bugs with conversation history
- Suggested improvements welcome
- Mode-specific feedback valuable
- Safety concern reporting critical

---

**Version:** 1.0 | **Updated:** 2026-03-27 | **Status:** Production-Ready ✅

