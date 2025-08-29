from enum import Enum
from fastapi import APIRouter, HTTPException, status, Request, BackgroundTasks, File, UploadFile, Form
from typing import List, Dict, Any, Optional
from app.services.research_agent import ResearchAgent
from pydantic import BaseModel
import asyncio
import logging
import time
import json
import traceback
import os
import tempfile
import PyPDF2
import docx
import shutil
from starlette.responses import JSONResponse

router = APIRouter(prefix="/research", tags=["research"])
logger = logging.getLogger(__name__)

class ResearchRequest(BaseModel):
    """Request model for research endpoint.
    
    Attributes:
        query: The research question or topic to investigate
        chat_history: Optional list of previous messages for context
    """
    query: str
    chat_history: Optional[List[Dict[str, Any]]] = None
    file_content: Optional[str] = None
    file_name: Optional[str] = None

class ResearchStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class ResearchStep(str, Enum):
    QUERY_ANALYSIS = "query_analysis"
    SOURCE_RETRIEVAL = "source_retrieval"
    DOCUMENT_PROCESSING = "document_processing"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"

class ResearchResponse(BaseModel):
    success: bool
    status: ResearchStatus
    step: Optional[ResearchStep] = None
    message: Optional[str] = None
    response: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

# In-memory store for research status (in production, use a proper cache like Redis)
research_status = {}

# Track cancellation requests
cancelled_requests = set()

def extract_text_from_file(file_path: str, file_extension: str) -> str:
    """Extract text from uploaded files based on their extension."""
    try:
        if file_extension.lower() == ".pdf":
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text() + "\n"
                return text
                
        elif file_extension.lower() in [".docx", ".doc"]:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
            
        elif file_extension.lower() == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
                
        else:
            return f"Unsupported file format: {file_extension}"
    except Exception as e:
        logger.error(f"Error extracting text from file: {str(e)}", exc_info=True)
        return f"Error extracting text: {str(e)}"

@router.post("", response_model=ResearchResponse)
async def research_endpoint(
    research_request: ResearchRequest,
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Handle research queries using the AI research assistant.
    
    This endpoint initiates a background task for research and returns a task ID
    that can be used to check the status and retrieve results.
    
    Args:
        research_request: ResearchRequest containing query and optional chat history
        request: FastAPI request object for logging
        
    Returns:
        ResearchResponse with initial status and task ID
    """
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    # Log request details
    logger.info(f"[{request_id}] Received research request")
    logger.info(f"[{request_id}] Query: {research_request.query}")
    
    # Initialize research status
    research_status[request_id] = {
        "status": ResearchStatus.PROCESSING,
        "step": ResearchStep.QUERY_ANALYSIS,
        "start_time": start_time,
        "last_updated": time.time(),
        "progress": 0,
        "message": "Starting research...",
        "result": None,
        "error": None
    }
    
    # Start the research in the background
    background_tasks.add_task(
        process_research,
        request_id=request_id,
        query=research_request.query,
        chat_history=research_request.chat_history or []
    )
    
    return ResearchResponse(
        success=True,
        status=ResearchStatus.PROCESSING,
        step=ResearchStep.QUERY_ANALYSIS,
        message="Research started. Use the task ID to check status.",
        metadata={
            "request_id": request_id,
            "status_url": f"/api/v1/research/status/{request_id}"
        }
    )

@router.post("/upload", response_model=ResearchResponse)
async def research_with_file(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    query: str = Form(...),
    chat_history: Optional[str] = Form(None)
):
    """Handle research queries with file uploads."""
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    # Log request details
    logger.info(f"[{request_id}] Received research request with file upload")
    logger.info(f"[{request_id}] Query: {query}")
    logger.info(f"[{request_id}] File: {file.filename}")
    
    # Create a temporary directory to store the uploaded file
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Extract text from the file based on its extension
        file_extension = os.path.splitext(file.filename)[1]
        file_content = extract_text_from_file(file_path, file_extension)
        
        # Parse chat history if provided
        parsed_chat_history = []
        if chat_history:
            try:
                parsed_chat_history = json.loads(chat_history)
            except json.JSONDecodeError:
                logger.warning(f"[{request_id}] Invalid chat history format")
        
        # Create a research request with the extracted text
        research_req = ResearchRequest(
            query=query,
            chat_history=parsed_chat_history,
            file_content=file_content,
            file_name=file.filename
        )
        
        # Initialize research status
        research_status[request_id] = {
            "status": ResearchStatus.PROCESSING,
            "step": ResearchStep.QUERY_ANALYSIS,
            "start_time": start_time,
            "last_updated": time.time(),
            "progress": 0,
            "message": "Starting research with uploaded file...",
            "result": None,
            "error": None
        }
        
        # Start the research in the background
        background_tasks.add_task(
            process_research,
            request_id=request_id,
            query=query,
            chat_history=parsed_chat_history,
            file_content=file_content,
            file_name=file.filename
        )
        
        return ResearchResponse(
            success=True,
            status=ResearchStatus.PROCESSING,
            step=ResearchStep.QUERY_ANALYSIS,
            message="Research with file upload started. Use the task ID to check status.",
            metadata={
                "request_id": request_id,
                "status_url": f"/api/v1/research/status/{request_id}",
                "file_name": file.filename
            }
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Error processing file upload: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file upload: {str(e)}"
        )
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

@router.post("/text", response_model=ResearchResponse)
async def research_endpoint(
    research_request: ResearchRequest,
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Handle research queries using the AI research assistant.
    
    This endpoint initiates a background task for research and returns a task ID
    that can be used to check the status and retrieve results.
    
    Args:
        research_request: ResearchRequest containing query and optional chat history
        request: FastAPI request object for logging
        
    Returns:
        ResearchResponse with initial status and task ID
    """
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    # Log request details
    logger.info(f"[{request_id}] Received research request")
    logger.info(f"[{request_id}] Query: {research_request.query}")
    
    # Initialize research status
    research_status[request_id] = {
        "status": ResearchStatus.PROCESSING,
        "step": ResearchStep.QUERY_ANALYSIS,
        "start_time": start_time,
        "last_updated": time.time(),
        "progress": 0,
        "message": "Starting research...",
        "result": None,
        "error": None
    }
    
    # Start the research in the background
    background_tasks.add_task(
        process_research,
        request_id=request_id,
        query=research_request.query,
        chat_history=research_request.chat_history or []
    )
    
    return ResearchResponse(
        success=True,
        status=ResearchStatus.PROCESSING,
        step=ResearchStep.QUERY_ANALYSIS,
        message="Research started. Use the task ID to check status.",
        metadata={
            "request_id": request_id,
            "status_url": f"/api/v1/research/status/{request_id}"
        }
    )

@router.get("/status/{request_id}", response_model=ResearchResponse)
async def get_research_status(request_id: str):
    """Get the status of a research request."""
    if request_id not in research_status:
        raise HTTPException(status_code=404, detail="Research request not found")
    
    status = research_status[request_id]
    
    # Check if this request has been cancelled
    if request_id in cancelled_requests:
        status["status"] = ResearchStatus.ERROR
        status["message"] = "Research was cancelled by user"
        status["error"] = {"message": "Research was cancelled by user", "type": "UserCancellation"}
    
    return ResearchResponse(
        success=status["status"] == ResearchStatus.COMPLETED,
        status=status["status"],
        step=status.get("step"),
        message=status.get("message"),
        response=status.get("result"),
        metadata={
            "progress": status.get("progress", 0),
            "start_time": status.get("start_time"),
            "elapsed_time": time.time() - status.get("start_time", time.time()),
            "cancelled": request_id in cancelled_requests,
            "error": status.get("error")
        }
    )

@router.post("/cancel/{request_id}", response_model=ResearchResponse)
async def cancel_research(request_id: str):
    """Cancel an ongoing research request."""
    if request_id not in research_status:
        raise HTTPException(status_code=404, detail="Research request not found")
    
    # Mark the request as cancelled
    cancelled_requests.add(request_id)
    
    # Update the research status
    status = research_status[request_id]
    status["status"] = ResearchStatus.ERROR
    status["message"] = "Research was cancelled by user"
    status["error"] = {"message": "Research was cancelled by user", "type": "UserCancellation"}
    
    return ResearchResponse(
        success=False,
        status=ResearchStatus.ERROR,
        message="Research was cancelled by user",
        metadata={
            "cancelled": True,
            "request_id": request_id
        }
    )

async def process_research(request_id: str, query: str, chat_history: List[Dict[str, Any]], file_content: str = None, file_name: str = None):
    """Process the research request in the background."""
    start_time = time.time()
    research_agent = ResearchAgent()
    
    # Check if this request has been cancelled before starting
    if request_id in cancelled_requests:
        logger.info(f"[{request_id}] Research was cancelled before processing started")
        research_status[request_id].update({
            "status": ResearchStatus.ERROR,
            "message": "Research was cancelled by user",
            "error": {"message": "Research was cancelled by user", "type": "UserCancellation"}
        })
        return
    
    def update_status(step: ResearchStep, message: str, progress: int):
        """Helper to update research status with validation and logging."""
        try:
            # Check if this request has been cancelled
            if request_id in cancelled_requests:
                logger.info(f"[{request_id}] Detected cancellation during update_status")
                if request_id in research_status:
                    research_status[request_id].update({
                        "status": ResearchStatus.ERROR,
                        "message": "Research was cancelled by user",
                        "error": {"message": "Research was cancelled by user", "type": "UserCancellation"},
                        "last_updated": time.time()
                    })
                return True  # Return True to indicate cancellation
            
            if not request_id or not isinstance(request_id, str):
                logger.error(f"[update_status] Invalid request_id: {request_id}")
                return False
                
            if not isinstance(step, ResearchStep):
                logger.error(f"[update_status] Invalid step: {step}")
                return False
                
            if not isinstance(progress, int) or not (0 <= progress <= 100):
                logger.error(f"[update_status] Invalid progress: {progress}")
                progress = max(0, min(100, int(progress or 0)))
                
            if request_id not in research_status:
                logger.error(f"[update_status] Request ID not found: {request_id}")
                return False
                
            # Only update if there's an actual change
            current = research_status[request_id]
            if (current.get('step') == step and 
                current.get('message') == message and 
                current.get('progress') == progress):
                return False
                
            # Update the status
            research_status[request_id].update({
                "step": step,
                "message": message,
                "progress": progress,
                "last_updated": time.time()
            })
            logger.info(f"[{request_id}] {step.value.upper()}: {message} (Progress: {progress}%)")
            return False  # Return False to indicate no cancellation
            
        except Exception as e:
            logger.error(f"[update_status] Error updating status: {str(e)}", exc_info=True)
            return False
    
    try:
        logger.info(f"[{request_id}] Starting research for query: {query[:100]}...")
        
        # Initialize research status
        research_status[request_id] = {
            "status": ResearchStatus.PROCESSING,
            "step": ResearchStep.QUERY_ANALYSIS,
            "start_time": start_time,
            "last_updated": time.time(),
            "progress": 0,
            "message": "Starting research...",
            "result": None,
            "error": None
        }
        
        # Step 1: Query Analysis
        is_cancelled = update_status(
            step=ResearchStep.QUERY_ANALYSIS,
            message="Analyzing your research question",
            progress=10
        )
        
        if is_cancelled:
            logger.info(f"[{request_id}] Research cancelled during query analysis")
            return
        
        # Execute the research with timeout
        try:
            # Include file content in the research if available
            research_kwargs = {
                "query": query,
                "chat_history": chat_history,
                # Add a cancellation check function
                "cancellation_check": lambda: request_id in cancelled_requests
            }
            
            # Add file content to the research if available
            if file_content and file_name:
                logger.info(f"[{request_id}] Including file content from {file_name} in research")
                # Add file content to the query for context
                enhanced_query = f"Research query: {query}\n\nAdditional context from file '{file_name}':\n{file_content[:2000]}..."
                research_kwargs["query"] = enhanced_query
            
            # Check for cancellation before starting research
            if request_id in cancelled_requests:
                logger.info(f"[{request_id}] Research cancelled before starting research agent")
                research_status[request_id].update({
                    "status": ResearchStatus.ERROR,
                    "message": "Research was cancelled by user",
                    "error": {"message": "Research was cancelled by user", "type": "UserCancellation"},
                    "last_updated": time.time()
                })
                return
                
            research_data = await asyncio.wait_for(
                research_agent.research(**research_kwargs),
                timeout=300  # 5 minutes timeout
            )
            
            if not research_data:
                raise ValueError("No research data returned from the research agent")
                
        except asyncio.TimeoutError:
            raise TimeoutError("Research operation timed out after 5 minutes")
        except Exception as e:
            logger.error(f"[{request_id}] Error in research agent: {str(e)}", exc_info=True)
            raise
            
        # Update progress through each step
        steps = [
            (ResearchStep.SOURCE_RETRIEVAL, "Searching for relevant sources", 30),
            (ResearchStep.DOCUMENT_PROCESSING, "Processing documents", 50),
            (ResearchStep.ANALYSIS, "Analyzing the gathered information", 70),
            (ResearchStep.SYNTHESIS, "Synthesizing the results", 90)
        ]
        
        for step, message, progress in steps:
            is_cancelled = update_status(step, message, progress)
            if is_cancelled:
                logger.info(f"[{request_id}] Research cancelled during {step.value}")
                return
            
            # Check for cancellation before sleeping
            if request_id in cancelled_requests:
                logger.info(f"[{request_id}] Research cancelled during {step.value} sleep")
                research_status[request_id].update({
                    "status": ResearchStatus.ERROR,
                    "message": "Research was cancelled by user",
                    "error": {"message": "Research was cancelled by user", "type": "UserCancellation"},
                    "last_updated": time.time()
                })
                return
                
            # Simulate some processing time between steps
            await asyncio.sleep(0.5)
        
        # Prepare and validate the response
        try:
            if not isinstance(research_data, dict):
                raise ValueError("Research data is not a dictionary")
                
            # Ensure required fields exist
            response = {
                "success": research_data.get("success", True),
                "status": "completed",
                "response": research_data.get("response") or "No response generated",
                "sources": research_data.get("sources", []),
                "recommendations": research_data.get("recommendations", []),
                "evaluation": research_data.get("evaluation", []),
                "metadata": {
                    "queryTime": time.time() - start_time,
                    "modelUsed": research_data.get("metadata", {}).get("model_used") or "default",
                    "tokenCount": research_data.get("metadata", {}).get("token_count", 0),
                    "requestId": request_id,
                    "timestamp": time.time(),
                    "stepsCompleted": [s[0].value for s in steps]  # Track completed steps
                }
            }
            
            # Log response summary
            logger.info(f"[{request_id}] Research response prepared successfully")
            logger.debug(f"[{request_id}] Response keys: {list(response.keys())}")
            
        except Exception as e:
            logger.error(f"[{request_id}] Error preparing response: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to prepare research response: {str(e)}")
        
        # Validate and format sources
        for i, source in enumerate(response["sources"] or []):
            if not isinstance(source, dict):
                response["sources"][i] = {
                    "title": "Unknown Source", 
                    "url": "#", 
                    "type": "unknown"
                }
            else:
                if "title" not in source:
                    source["title"] = source.get("content", "")[:100] or "Untitled Source"
                if "url" not in source:
                    source["url"] = "#"
                if "type" not in source:
                    source["type"] = "website"
        
        # Update status with results
        research_status[request_id].update({
            "status": ResearchStatus.COMPLETED,
            "result": response,
            "progress": 100,
            "message": "Research completed successfully",
            "last_updated": time.time()
        })
        
        logger.info(f"[{request_id}] Research completed successfully in {time.time() - start_time:.2f} seconds")
        return response
        
    except Exception as e:
        error_msg = f"Research failed: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}", exc_info=True)
        
        if request_id in research_status:
            research_status[request_id].update({
                "status": ResearchStatus.ERROR,
                "error": {
                    "message": str(e),
                    "type": e.__class__.__name__,
                    "traceback": traceback.format_exc()
                },
                "message": f"Research failed: {str(e)}",
                "last_updated": time.time()
            })
        
        # Re-raise to ensure the error is logged by the error handler
        raise

# Add more endpoints as needed, such as for specific research tasks or evaluations
