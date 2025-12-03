import os
import asyncio
from typing import Dict, Any, List
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configure API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_API_KEY not found in .env file. The agent will fail.")
else:
    genai.configure(api_key=api_key)

try:
    from .memory import MedicalProfile, Symptom
except ImportError:
    from memory import MedicalProfile, Symptom

# Import modules
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

_current_profile: MedicalProfile = None
_handoff_triggered: bool = False

def update_medical_profile(
    main_complaint: str = None,
    symptom: str = None,
    duration: str = None,
    severity: str = None,
    medication: str = None,
    family_history: str = None
) -> dict:
    """
    Updates the patient's medical profile with new information.
    Use this tool whenever the patient mentions symptoms, medications, or medical history.
    
    Args:
        main_complaint: The primary reason for the visit (e.g., "chest pain", "headache").
        symptom: A specific symptom to record (e.g., "nausea", "dizziness").
        duration: How long the symptom has been present (e.g., "2 days", "1 week").
        severity: How severe the symptom is (e.g., "mild", "moderate", "severe", "7/10").
        medication: A medication the patient is currently taking.
        family_history: A relevant family medical history item.
    """
    global _current_profile
    
    updates = []
    
    if main_complaint and _current_profile:
        _current_profile.main_complaint = main_complaint
        updates.append(f"main_complaint: {main_complaint}")
    
    if symptom and _current_profile:
        existing = next((s for s in _current_profile.symptoms if s.description.lower() == symptom.lower()), None)
        if existing:
            if duration: existing.duration = duration
            if severity: existing.severity = severity
        else:
            _current_profile.symptoms.append(Symptom(
                description=symptom,
                duration=duration or "Unknown",
                severity=severity or "Unknown"
            ))
        updates.append(f"symptom: {symptom}")
    
    if medication and _current_profile and medication not in _current_profile.medications:
        _current_profile.medications.append(medication)
        updates.append(f"medication: {medication}")
    
    if family_history and _current_profile and family_history not in _current_profile.family_history:
        _current_profile.family_history.append(family_history)
        updates.append(f"family_history: {family_history}")
    
    return {"status": "updated", "updates": updates}

def complete_interview() -> dict:
    """
    Call this function ONLY when you have gathered sufficient medical information 
    (main complaint + at least 2-3 symptoms/details) and are ready to hand off
    to the Medical Analyst for report generation.
    """
    global _handoff_triggered
    _handoff_triggered = True
    return {"status": "handoff", "message": "Interview complete. Handing off to Medical Analyst."}

def web_search(query: str) -> dict:
    """
    Search the web for medical information related to the query.
    Use this to research unfamiliar symptoms or conditions.
    
    Args:
        query: The search query for medical information.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            summaries = [f"- {r['title']}: {r['body']}" for r in results]
            return {"status": "success", "results": "\n".join(summaries)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def submit_doctor_questions(question1: str, question2: str, question3: str, question4: str = None, question5: str = None) -> dict:
    """
    Submits the suggested questions for the doctor to ask during the visit.
    Use this after analyzing the patient profile to generate relevant clinical questions.
    
    Args:
        question1: First important question for the doctor.
        question2: Second important question for the doctor.
        question3: Third important question for the doctor.
        question4: Optional fourth question for the doctor.
        question5: Optional fifth question for the doctor.
    """
    global _current_profile
    
    questions = [q for q in [question1, question2, question3, question4, question5] if q]
    
    if _current_profile and questions:
        _current_profile.suggested_questions = questions
    
    return {"status": "submitted", "questions": questions}

# ============================================================================
# AI AGENTS
# ============================================================================

interviewer_instruction = """You are 'Prepped', a warm and professional medical intake advocate. 
Your role is to conduct a structured interview to prepare the patient for a doctor's visit.

CRITICAL RULES:
1. Do NOT diagnose or suggest medications.
2. If user mentions emergency symptoms (chest pain, trouble breathing, severe bleeding), tell them to call 911 immediately.
3. Ask ONE question at a time. Be conversational and empathetic.
4. Use the 'update_medical_profile' tool to record ANY medical information the patient shares.
5. When you have gathered the main complaint AND at least 2-3 supporting details (symptoms, duration, severity, medications), call 'complete_interview' function.

INTERVIEW FLOW:
1. Start: "What brings you in today?"
2. Gather: Main complaint, symptom details, duration, severity
3. Ask about: Current medications, relevant family history
4. Complete: Call complete_interview() when ready

Always be supportive and acknowledge the patient's concerns."""

analyst_instruction = """You are a Medical Analyst AI. Your job is to review patient intake data and prepare a clinical briefing for the doctor.

TASK:
1. Review the patient's medical profile provided in context.
2. Use 'web_search' to research any unfamiliar symptoms or conditions if needed.
3. Generate a structured briefing in Markdown format.
4. Call 'submit_doctor_questions' with 3-5 relevant clinical questions.

OUTPUT FORMAT:
# Doctor Briefing

**Chief Complaint:** [Main reason for visit]

**Symptom Summary:**
- [Symptom 1]: Duration, Severity
- [Symptom 2]: Duration, Severity

**Current Medications:** [List or "None reported"]

**Family History:** [List or "None reported"]

**Clinical Considerations:**
- [Relevant observations]
- [Potential red flags]

**Suggested Questions for Doctor:**
1. [Question 1]
2. [Question 2]
3. [Question 3]

Focus on clinical differentiators and red flags. Do NOT ask questions about information already in the profile."""

intake_nurse_agent = Agent(
    name="intake_nurse",
    model="gemini-2.5-flash",
    description="A medical intake nurse that interviews patients to gather health information.",
    instruction=interviewer_instruction,
    tools=[update_medical_profile, complete_interview],
)

medical_analyst_agent = Agent(
    name="medical_analyst", 
    model="gemini-2.5-flash",
    description="A medical analyst that reviews patient data and prepares doctor briefings.",
    instruction=analyst_instruction,
    tools=[web_search, submit_doctor_questions],
)

session_service = InMemorySessionService()
_runners: Dict[str, Runner] = {}

def get_runner(agent_type: str) -> Runner:
    """Get or create a runner for the specified agent type."""
    if agent_type not in _runners:
        agent = intake_nurse_agent if agent_type == "interviewer" else medical_analyst_agent
        _runners[agent_type] = Runner(
            agent=agent,
            app_name="prepped_medical_intake",
            session_service=session_service
        )
    return _runners[agent_type]

async def run_agent_async(
    agent_type: str,
    user_input: str, 
    session_id: str, 
    profile: MedicalProfile
) -> Dict[str, Any]:
    global _current_profile, _handoff_triggered
    
    _current_profile = profile
    _handoff_triggered = False
    
    runner = get_runner(agent_type)
    steps = []
    final_response = ""
    
    user_id = f"user_{session_id}"
    adk_session_id = f"{agent_type}_{session_id}"
    
    existing_session = await session_service.get_session(
        app_name="prepped_medical_intake",
        user_id=user_id,
        session_id=adk_session_id
    )
    
    if not existing_session:
        await session_service.create_session(
            app_name="prepped_medical_intake",
            user_id=user_id,
            session_id=adk_session_id
        )
    
    context_message = f"""
[CURRENT PATIENT PROFILE]
{profile.to_text()}
[END PROFILE]

Patient says: {user_input}
"""
    
    user_content = types.Content(
        role="user",
        parts=[types.Part(text=context_message)]
    )
    
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=adk_session_id,
            new_message=user_content
        ):
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts') and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            final_response = part.text
                        elif hasattr(part, 'function_call') and part.function_call:
                            fc = part.function_call
                            if fc and hasattr(fc, 'name') and fc.name:
                                args_str = ""
                                if hasattr(fc, 'args') and fc.args:
                                    args_str = str(dict(fc.args))
                                
                                # Add tool call step with full details
                                steps.append({
                                    "type": "tool_call",
                                    "content": f"{fc.name}({args_str})" if args_str else f"{fc.name}()"
                                })
                                
                                if fc.name == "update_medical_profile":
                                    steps.append({"type": "action", "content": "Updated Medical Profile"})
                                elif fc.name == "complete_interview":
                                    steps.append({"type": "handoff", "content": "Handing off to Medical Analyst"})
                                elif fc.name == "submit_doctor_questions":
                                    steps.append({"type": "action", "content": "Submitted Doctor Questions"})
                                
    except Exception as e:
        print(f"ADK Agent Error: {e}")
        import traceback
        traceback.print_exc()
        final_response = "I'm having trouble processing that. Could you please repeat?"
    
    if not final_response:
        if _handoff_triggered:
            final_response = "I have gathered enough information. I am now preparing your briefing for the doctor..."
        else:
            final_response = "I've noted that. Could you tell me more?"
    
    agent_name = "Intake Nurse" if agent_type == "interviewer" else "Medical Analyst"
    
    return {
        "response": final_response,
        "is_handoff": _handoff_triggered,
        "profile": profile,
        "agent_name": agent_name,
        "steps": steps
    }

def _get_or_create_event_loop():
    try:
        loop = asyncio.get_running_loop()
        return loop, False
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop, True

def run_interviewer(user_input: str, session_id: str, profile: MedicalProfile) -> Dict[str, Any]:
    loop, created = _get_or_create_event_loop()
    try:
        if created:
            return loop.run_until_complete(run_agent_async("interviewer", user_input, session_id, profile))
        else:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, run_agent_async("interviewer", user_input, session_id, profile))
                return future.result()
    finally:
        if created:
            loop.close()

def run_analyst(user_input: str, session_id: str, profile: MedicalProfile) -> Dict[str, Any]:
    loop, created = _get_or_create_event_loop()
    try:
        if created:
            return loop.run_until_complete(run_agent_async("analyst", user_input, session_id, profile))
        else:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, run_agent_async("analyst", user_input, session_id, profile))
                return future.result()
    finally:
        if created:
            loop.close()

def transcribe_audio(base64_audio: str) -> str:
    try:
        if "base64," in base64_audio:
            base64_audio = base64_audio.split("base64,")[1]
            
        import base64
        audio_bytes = base64.b64decode(base64_audio)
        
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        response = model.generate_content([
            "Please transcribe the following audio file verbatim. Do not add any commentary.",
            {
                "mime_type": "audio/mp3",
                "data": audio_bytes
            }
        ])
        return response.text.strip()
    except Exception as e:
        print(f"Transcription failed: {e}")
        return "[Audio Transcription Failed]"

class LegacyAgentWrapper:
    def __init__(self, name: str, agent_type: str):
        self.name = name
        self.agent_type = agent_type
    
    def run(self, user_input: str, session_id: str, profile: MedicalProfile) -> Dict[str, Any]:
        if self.agent_type == "interviewer":
            return run_interviewer(user_input, session_id, profile)
        else:
            return run_analyst(user_input, session_id, profile)

interviewer_agent = LegacyAgentWrapper("Intake Nurse", "interviewer")
analyst_agent = LegacyAgentWrapper("Medical Analyst", "analyst")