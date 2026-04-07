"""
Background Tasks Module
Handles slow operations asynchronously:
- GPT API calls
- PDF generation
- Data processing
- Email notifications

Uses in-memory queue (can be replaced with Celery/Redis in production)
"""

import asyncio
import logging
from typing import Callable, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from collections import deque
import uuid

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Background task representation"""
    id: str
    func: Callable
    args: tuple
    kwargs: dict
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class BackgroundTaskQueue:
    """
    In-memory task queue for background operations
    For production: replace with Celery + Redis
    """
    
    def __init__(self, max_workers: int = 4, max_queue_size: int = 1000):
        """
        Initialize task queue
        
        Args:
            max_workers: Number of concurrent workers
            max_queue_size: Maximum pending tasks
        """
        self.queue: deque = deque(maxlen=max_queue_size)
        self.tasks: dict[str, Task] = {}
        self.max_workers = max_workers
        self.active_workers = 0
        self.running = False
    
    async def add_task(self, func: Callable, *args, 
                      task_name: str = None, **kwargs) -> str:
        """
        Add task to queue
        
        Returns:
            Task ID for tracking
        """
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs
        )
        
        self.queue.append(task)
        self.tasks[task_id] = task
        
        logger.info(f"Task added: {task_name or func.__name__} (ID: {task_id})")
        
        # Process if workers available
        if self.active_workers < self.max_workers:
            asyncio.create_task(self._process_task(task))
        
        return task_id
    
    async def _process_task(self, task: Task):
        """Execute task with error handling"""
        self.active_workers += 1
        
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            logger.info(f"Task started: {task.id}")
            
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(task.func):
                result = await task.func(*task.args, **task.kwargs)
            else:
                # Run sync function in executor to prevent blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    lambda: task.func(*task.args, **task.kwargs)
                )
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            logger.info(f"Task completed: {task.id}")
        
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            
            logger.error(f"Task failed {task.id}: {str(e)}")
        
        finally:
            self.active_workers -= 1
            
            # Process next task if available
            if self.queue and self.active_workers < self.max_workers:
                next_task = self.queue.popleft()
                asyncio.create_task(self._process_task(next_task))
    
    def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get task status and result"""
        return self.tasks.get(task_id)
    
    def get_queue_stats(self) -> dict:
        """Get queue statistics"""
        completed = sum(1 for t in self.tasks.values() 
                       if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks.values() 
                    if t.status == TaskStatus.FAILED)
        
        return {
            "queue_size": len(self.queue),
            "total_tasks": len(self.tasks),
            "active_workers": self.active_workers,
            "completed": completed,
            "failed": failed,
            "pending": sum(1 for t in self.tasks.values() 
                         if t.status == TaskStatus.PENDING)
        }


# Global task queue instance
task_queue = BackgroundTaskQueue(max_workers=4)


# ============ ASYNC GPT CALL WRAPPER ============

async def call_gpt_async(prompt: str, system_message: str = None,
                        max_tokens: int = 500) -> str:
    """
    Async wrapper for GPT calls
    Prevent blocking main thread
    """
    from openai import OpenAI
    import os
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    try:
        # Run in executor to prevent blocking
        loop = asyncio.get_event_loop()
        
        def _call_gpt():
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        
        result = await loop.run_in_executor(None, _call_gpt)
        return result
    
    except Exception as e:
        logger.error(f"GPT call failed: {e}")
        return f"Error: {str(e)}"


async def queue_gpt_call(prompt: str, system_message: str = None) -> str:
    """
    Queue GPT call as background task
    Returns task ID immediately
    """
    task_id = await task_queue.add_task(
        call_gpt_async,
        prompt,
        system_message=system_message,
        task_name="gpt_call"
    )
    return task_id


# ============ ASYNC PDF GENERATION ============

async def generate_pdf_async(patient_data: dict, output_path: str) -> str:
    """
    Async PDF generation
    Prevents blocking on report creation
    """
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    
    try:
        loop = asyncio.get_event_loop()
        
        def _generate():
            doc = SimpleDocTemplate(output_path)
            styles = getSampleStyleSheet()
            story = []
            
            # Add content
            title = f"Clinical Report - {patient_data.get('name', 'Patient')}"
            story.append(Paragraph(title, styles['Heading1']))
            story.append(Spacer(1, 12))
            
            # Add patient data
            for key, value in patient_data.items():
                story.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
                story.append(Spacer(1, 6))
            
            doc.build(story)
            return output_path
        
        result = await loop.run_in_executor(None, _generate)
        return result
    
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise


async def queue_pdf_generation(patient_data: dict, output_path: str) -> str:
    """Queue PDF generation as background task"""
    task_id = await task_queue.add_task(
        generate_pdf_async,
        patient_data,
        output_path,
        task_name="pdf_generation"
    )
    return task_id


# ============ BATCH PROCESSING ============

async def batch_process_predictions(patient_list: List[dict],
                                   prediction_func: Callable) -> str:
    """
    Queue batch prediction processing
    Process multiple patients without blocking
    """
    async def _batch_process():
        results = []
        for patient in patient_list:
            try:
                result = await asyncio.coroutine(prediction_func)(patient)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                results.append({"error": str(e)})
        
        return results
    
    task_id = await task_queue.add_task(
        _batch_process,
        task_name="batch_predictions"
    )
    return task_id


# ============ PRODUCTION DEPLOYMENT READY CELERY STUB ============

"""
For production deployment with Celery + Redis:

from celery import Celery

app = Celery('hemophilia_clinic')
app.config_from_object('celery_config')

@app.task(bind=True, max_retries=3)
def gpt_call_task(self, prompt: str, system_message: str = None):
    '''Celery task for async GPT calls'''
    try:
        result = call_gpt_async(prompt, system_message)
        return result
    except Exception as exc:
        self.retry(exc=exc, countdown=60)

@app.task(bind=True, max_retries=2)
def pdf_generation_task(self, patient_data: dict, output_path: str):
    '''Celery task for async PDF generation'''
    try:
        result = generate_pdf_async(patient_data, output_path)
        return result
    except Exception as exc:
        self.retry(exc=exc, countdown=30)

Deployment config (celery_config.py):

broker_url = 'redis://localhost:6379'
result_backend = 'redis://localhost:6379'
task_serializer = 'json'
accept_content = ['json']
result_serializer = 'json'
timezone = 'UTC'
enable_utc = True
task_track_started = True
task_soft_time_limit = 900
task_time_limit = 1200
"""
