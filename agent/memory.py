import json
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.id import ID
from appwrite.exception import AppwriteException
from dotenv import load_dotenv
import warnings

load_dotenv()

APPWRITE_ENDPOINT = os.getenv("APPWRITE_ENDPOINT")
APPWRITE_PROJECT_ID = os.getenv("APPWRITE_PROJECT_ID")
APPWRITE_API_KEY = os.getenv("APPWRITE_API_KEY")
APPWRITE_DATABASE_ID = os.getenv("APPWRITE_DATABASE_ID")
APPWRITE_COLLECTION_ID = os.getenv("APPWRITE_COLLECTION_ID")

warnings.filterwarnings("ignore", category=DeprecationWarning, module="appwrite")

client = Client()
client.set_endpoint(APPWRITE_ENDPOINT)
client.set_project(APPWRITE_PROJECT_ID)
client.set_key(APPWRITE_API_KEY)

databases = Databases(client)

class Symptom(BaseModel):
    description: str
    duration: str = "Unknown"
    severity: str = "Unknown"

class ChatMessage(BaseModel):
    id: str
    role: str
    content: str
    type: str = "text"
    timestamp: int
    agent_name: Optional[str] = None
    trace: Optional[List[Dict[str, Any]]] = None

class MedicalProfile(BaseModel):
    """
    A structured representation of the patient's medical context.
    This acts as the 'Memory Bank' for the agentic system.
    """
    status: str = "interview"
    user_id: Optional[str] = None
    call_status: Optional[str] = None 
    call_sid: Optional[str] = None
    session_title: Optional[str] = None
    main_complaint: Optional[str] = None
    symptoms: List[Symptom] = []
    medications: List[str] = []
    family_history: List[str] = []
    suggested_questions: List[str] = []
    messages: List[ChatMessage] = []
    
    def to_text(self) -> str:
        """
        Converts the profile into a natural language string for the LLM context.
        """
        symptoms_text = "None recorded"
        if self.symptoms:
            symptoms_text = "; ".join([f"{s.description} (Duration: {s.duration}, Severity: {s.severity})" for s in self.symptoms])

        return f"""
        Current Medical Profile:
        - Status: {self.status}
        - Main Complaint: {self.main_complaint or "Unknown"}
        - Symptoms: {symptoms_text}
        - Medications: {", ".join(self.medications) if self.medications else "None recorded"}
        - Family History: {", ".join(self.family_history) if self.family_history else "None recorded"}
        """

def save_session(session_id: str, profile: MedicalProfile):
    """Saves the session to Appwrite."""
    
    data_json = profile.model_dump_json()
    
    def _save(is_create=False):
        payload = {'data': [data_json]}
        
        try:
            if is_create:
                databases.create_document(
                    database_id=APPWRITE_DATABASE_ID,
                    collection_id=APPWRITE_COLLECTION_ID,
                    document_id=session_id,
                    data=payload
                )
            else:
                databases.update_document(
                    database_id=APPWRITE_DATABASE_ID,
                    collection_id=APPWRITE_COLLECTION_ID,
                    document_id=session_id,
                    data=payload
                )
        except AppwriteException as e:
            if "must be a string" in str(e).lower():
                print(f"Schema mismatch detected. Retrying with string format for session {session_id}...")
                payload_str = {'data': data_json}
                if is_create:
                    databases.create_document(
                        database_id=APPWRITE_DATABASE_ID,
                        collection_id=APPWRITE_COLLECTION_ID,
                        document_id=session_id,
                        data=payload_str
                    )
                else:
                    databases.update_document(
                        database_id=APPWRITE_DATABASE_ID,
                        collection_id=APPWRITE_COLLECTION_ID,
                        document_id=session_id,
                        data=payload_str
                    )
            else:
                raise e

    try:
        _save(is_create=False)
    except AppwriteException as e:
        if e.code == 404:
            try:
                _save(is_create=True)
            except Exception as create_error:
                print(f"Error creating session in Appwrite: {create_error}")
        else:
            print(f"Error saving session to Appwrite: {e}")

def get_session(session_id: str, auto_create: bool = True) -> MedicalProfile:
    """Retrieves or creates a session, loading from Appwrite if available.
    
    Args:
        session_id: The unique session identifier
        auto_create: If True, creates and saves a new session if not found.
                    If False, returns a new profile without saving (caller should save with user_id)
    """
    try:
        doc = databases.get_document(
            database_id=APPWRITE_DATABASE_ID,
            collection_id=APPWRITE_COLLECTION_ID,
            document_id=session_id
        )
        
        if 'data' in doc:
            raw_data = doc['data']
            if isinstance(raw_data, list):
                raw_data = raw_data[0] if raw_data else "{}"
                
            profile = MedicalProfile.model_validate_json(raw_data)
            return profile
            
    except AppwriteException as e:
        if e.code != 404:
            print(f"Appwrite error fetching session {session_id}: {e}")
    except Exception as e:
        print(f"Error parsing session data: {e}")
    
    new_profile = MedicalProfile()

    if auto_create:
        save_session(session_id, new_profile)
    
    return new_profile

def list_sessions(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lists all available sessions from Appwrite, optionally filtered by user_id."""
    from appwrite.query import Query
    
    try:
        result = databases.list_documents(
            database_id=APPWRITE_DATABASE_ID,
            collection_id=APPWRITE_COLLECTION_ID,
            queries=[
                Query.order_desc('$createdAt'),
                Query.limit(100)
            ]
        )
        
        sessions = []
        for doc in result['documents']:
            try:
                if 'data' in doc:
                    raw_data = doc['data']
                    if isinstance(raw_data, list):
                        raw_data = raw_data[0] if raw_data else "{}"
                        
                    profile_data = json.loads(raw_data)
                    doc_user_id = profile_data.get('user_id')
                    
                    if user_id and doc_user_id != user_id:
                        continue

                    title = profile_data.get('session_title') or profile_data.get('main_complaint') or "New Intake"
                    
                    sessions.append({
                        "id": doc['$id'],
                        "title": title,
                        "date": "Today",
                        "preview": f"Status: {profile_data.get('status', 'unknown')}",
                        "status": profile_data.get('status', 'interview')
                    })
            except Exception as e:
                print(f"Error parsing session {doc['$id']}: {e}")
        
        return sessions
    except AppwriteException as e:
        print(f"Appwrite error listing sessions: {e}")
        return []

def delete_session(session_id: str) -> bool:
    """Deletes a session from Appwrite."""
    try:
        databases.delete_document(
            database_id=APPWRITE_DATABASE_ID,
            collection_id=APPWRITE_COLLECTION_ID,
            document_id=session_id
        )
        return True
    except AppwriteException as e:
        print(f"Appwrite error deleting session {session_id}: {e}")
        return False

def generate_session_title(profile: MedicalProfile) -> str:
    """Generates a concise, descriptive title for a session using AI."""
    import google.generativeai as genai
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return profile.main_complaint or "Medical Intake"
    
    genai.configure(api_key=api_key)
    
    context_parts = []
    if profile.main_complaint:
        context_parts.append(f"Chief complaint: {profile.main_complaint}")
    if profile.symptoms:
        symptoms = ", ".join([s.description for s in profile.symptoms[:3]])
        context_parts.append(f"Symptoms: {symptoms}")
    
    if not context_parts:
        return "New Intake Session"
    
    context = ". ".join(context_parts)
    
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            f"Generate a very short (2-4 words max) medical session title for: {context}. "
            f"Examples: 'Chest Pain Review', 'Headache Consultation', 'Back Pain Intake'. "
            f"Just output the title, nothing else."
        )
        title = response.text.strip().strip('"').strip("'")
        if len(title) > 40:
            title = title[:37] + "..."
        return title
    except Exception as e:
        print(f"Error generating title: {e}")
        return profile.main_complaint or "Medical Intake"