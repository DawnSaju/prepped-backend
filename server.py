from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import os
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse

load_dotenv()

from agent.memory import get_session, save_session, list_sessions, delete_session, generate_session_title, ChatMessage
from agent.agent_logic import interviewer_agent, analyst_agent, transcribe_audio
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
BASE_URL = os.getenv("BASE_URL") 

@app.get("/")
async def root():
    return {"status": 200, "service": "prepped-backend"}

class ChatRequest(BaseModel):
    session_id: str
    message: str
    audio: Optional[str] = None
    user_id: Optional[str] = None

class CallRequest(BaseModel):
    session_id: str
    phone_number: str

class ChatResponse(BaseModel):
    response: str
    is_handoff: bool
    current_profile: Dict[str, Any]
    agent_name: str
    trace: List[Dict[str, str]] = []

@app.post("/initiate-call")
async def initiate_call(req: CallRequest):
    """
    Initiates an outbound call to the user's phone.
    """
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_PHONE_NUMBER:
        raise HTTPException(status_code=500, detail="Twilio credentials not configured.")
    
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        base_url_clean = BASE_URL.rstrip('/')
        webhook_url = f"{base_url_clean}/voice/webhook?session_id={req.session_id}"
        status_callback_url = f"{base_url_clean}/voice/status?session_id={req.session_id}"
        
        print(f"Initiating call to {req.phone_number} with webhook: {webhook_url}")
        
        call = client.calls.create(
            to=req.phone_number,
            from_=TWILIO_PHONE_NUMBER,
            url=webhook_url,
            status_callback=status_callback_url,
            status_callback_event=['initiated', 'ringing', 'answered', 'completed', 'busy', 'failed', 'no-answer']
        )
        
        memory = get_session(req.session_id)
        memory.call_status = "queued"
        memory.call_sid = call.sid
        save_session(req.session_id, memory)
        
        return {"status": "initiated", "call_sid": call.sid}
    except Exception as e:
        print(f"Twilio Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice/webhook")
async def voice_webhook(request: Request):
    """
    Handles the TwiML logic for the phone call.
    Twilio sends form data (SpeechResult, etc.) to this endpoint.
    """
    form_data = await request.form()
    session_id = request.query_params.get("session_id")
    
    user_speech = form_data.get("SpeechResult")
    
    resp = VoiceResponse()
    
    if not session_id:
        resp.say("System error. No session ID provided.")
        resp.hangup()
        return Response(content=str(resp), media_type="application/xml")

    memory = get_session(session_id)
    
    if not user_speech:
        intro = "Hello. This is Prepped, your medical intake assistant. I'm here to help you prepare for your doctor's visit. What brings you in today?"
        resp.say(intro, voice="Polly.Joanna-Neural")
        resp.gather(input="speech", action=f"/voice/webhook?session_id={session_id}", timeout=3)
        return Response(content=str(resp), media_type="application/xml")
    
    # print(f"Phone Input: {user_speech}")
    
    if memory.status == "interview":
        result = interviewer_agent.run(user_speech, session_id, memory)
        
        agent_response = result["response"]
        
        if result["is_handoff"]:
            memory.status = "analysis"
            
            handoff_prompt = "The interview is complete. Please review the profile and generate the Doctor Briefing."
            analyst_agent.run(handoff_prompt, session_id, memory)
            save_session(session_id, memory)
            
            goodbye = "Thank you. I have gathered enough information. Your doctor briefing is now ready and available on your dashboard. Goodbye."
            resp.say(goodbye, voice="Polly.Joanna-Neural")
            resp.hangup()
        else:
            save_session(session_id, memory)
            resp.say(agent_response, voice="Polly.Joanna-Neural")
            resp.gather(input="speech", action=f"/voice/webhook?session_id={session_id}", timeout=3)
            
    else:
        resp.say("This interview is already complete. Please check your dashboard.", voice="Polly.Joanna-Neural")
        resp.hangup()

    return Response(content=str(resp), media_type="application/xml")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    # Get session without auto-creating to allow setting user_id first
    memory = get_session(req.session_id, auto_create=False)
    
    # Set user_id before first save
    if req.user_id and not memory.user_id:
        memory.user_id = req.user_id
    
    # Save the session (this creates it if it doesn't exist, with user_id included)
    save_session(req.session_id, memory)
    
    user_message = req.message
    if req.audio:
        # print("Processing audio input...")
        transcription = transcribe_audio(req.audio)
        # print(f"Transcribed: {transcription}")
        if user_message:
            user_message = f"{user_message} {transcription}"
        else:
            user_message = transcription
    
    user_msg_obj = ChatMessage(
        id=str(int(time.time() * 1000)),
        role="user",
        content=user_message,
        timestamp=int(time.time() * 1000)
    )
    memory.messages.append(user_msg_obj)
            
    if memory.status == "interview":
        # Run Agent A (Intake Nurse)
        result = interviewer_agent.run(user_message, req.session_id, memory)
        
        if result["is_handoff"]:
            memory.status = "analysis"
            
            # Generate AI title for the session
            if not memory.session_title:
                memory.session_title = generate_session_title(memory)
            
            handoff_prompt = "The interview is complete. Please review the profile and generate the Doctor Briefing."
            result_b = analyst_agent.run(handoff_prompt, req.session_id, memory)            
            combined_response = f"{result['response']}\n\n---\n\n**Medical Analyst:**\n{result_b['response']}"            
            combined_trace = result.get("steps", []) + result_b.get("steps", [])
            
            agent_msg_obj = ChatMessage(
                id=str(int(time.time() * 1000) + 1),
                role="assistant",
                content=combined_response,
                timestamp=int(time.time() * 1000) + 1,
                agent_name="Medical Analyst",
                trace=combined_trace
            )
            memory.messages.append(agent_msg_obj)
            
            save_session(req.session_id, memory)
            return ChatResponse(
                response=combined_response,
                is_handoff=True,
                current_profile=memory.model_dump(),
                agent_name="Medical Analyst",
                trace=combined_trace
            )
        else:
            agent_msg_obj = ChatMessage(
                id=str(int(time.time() * 1000) + 1),
                role="assistant",
                content=result["response"],
                timestamp=int(time.time() * 1000) + 1,
                agent_name=result.get("agent_name", "Intake Nurse"),
                trace=result.get("steps", [])
            )
            memory.messages.append(agent_msg_obj)
            
            save_session(req.session_id, memory)
            return ChatResponse(
                response=result["response"],
                is_handoff=False,
                current_profile=memory.model_dump(),
                agent_name=result.get("agent_name", "Intake Nurse"),
                trace=result.get("steps", [])
            )
            
    elif memory.status == "analysis" or memory.status == "complete":
        # Run Agent B (Medical Analyst) for any follow-up questions
        result = analyst_agent.run(user_message, req.session_id, memory)
        
        agent_msg_obj = ChatMessage(
            id=str(int(time.time() * 1000) + 1),
            role="assistant",
            content=result["response"],
            timestamp=int(time.time() * 1000) + 1,
            agent_name="Medical Analyst",
            trace=result.get("steps", [])
        )
        memory.messages.append(agent_msg_obj)
        
        save_session(req.session_id, memory)
        return ChatResponse(
            response=result["response"],
            is_handoff=False,
            current_profile=memory.model_dump(),
            agent_name="Medical Analyst",
            trace=result.get("steps", [])
        )
    
    return ChatResponse(
        response="Session state error. Please reset.",
        is_handoff=False,
        current_profile=memory.model_dump(),
        agent_name="System",
        trace=[]
    )

@app.get("/session/{session_id}")
async def get_session_data(session_id: str):
    try:
        # Don't auto-create on GET - only return existing sessions
        # New sessions should be created via POST /chat with user_id
        memory = get_session(session_id, auto_create=False)
        return memory.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate-report/{session_id}")
async def generate_report(session_id: str):
    memory = get_session(session_id)
    profile_text = memory.to_text()
    
    # Run Agent B
    result = analyst_agent.run(profile_text, session_id, memory)
    memory.status = "complete"
    save_session(session_id, memory)
    
    return {"report_markdown": result["response"]}

@app.post("/voice/status")
async def voice_status(request: Request):
    """
    Handles Twilio status callbacks (ringing, answered, completed, etc.)
    """
    form_data = await request.form()
    session_id = request.query_params.get("session_id")
    call_status = form_data.get("CallStatus")
    
    # print(f"Call Status Update: {call_status} for session {session_id}")
    
    if session_id:
        memory = get_session(session_id)
        memory.call_status = call_status
        save_session(session_id, memory)
        
    return Response(status_code=200)

@app.get("/call-status/{session_id}")
async def get_call_status(session_id: str):
    """
    Endpoint for frontend polling to check call status.
    """
    try:
        memory = get_session(session_id)
        return {
            "call_status": memory.call_status,
            "status": memory.status,
            "call_sid": memory.call_sid
        }
    except Exception as e:
        return {"call_status": "unknown", "error": str(e)}

@app.get("/sessions")
async def get_all_sessions(user_id: Optional[str] = None):
    """
    Returns a list of all available chat sessions.
    """
    return list_sessions(user_id)

@app.delete("/session/{session_id}")
async def delete_session_endpoint(session_id: str):
    """
    Deletes a chat session.
    """
    success = delete_session(session_id)
    if success:
        return {"status": "deleted", "session_id": session_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete session")