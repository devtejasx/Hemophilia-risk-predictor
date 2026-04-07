"""
Gunicorn Configuration for Production Deployment

This file configures Gunicorn for optimal performance:
- Worker optimization
- Connection pooling
- Resource limits
- Logging and monitoring
- Production security
"""

import multiprocessing
import os
from datetime import datetime

# ============ WORKER CONFIGURATION ============

# Number of worker processes
# Formula: (2 x CPU cores) + 1
workers = int(os.environ.get("GUNICORN_WORKERS", (multiprocessing.cpu_count() * 2) + 1))

# Worker class: Using uvicorn workers for async support
worker_class = "uvicorn.workers.UvicornWorker"

# Threads per worker (for uvicorn)
threads = int(os.environ.get("GUNICORN_THREADS", 2))

# Worker connections/timeout
worker_connections = 1000
timeout = 120
keepalive = 5

# ============ PERFORMANCE TUNING ============

# Maximum requests per worker (restart to prevent memory leaks)
max_requests = 1000
max_requests_jitter = 100

# Server socket settings
backlog = 2048

# Bind address and port
bind = os.environ.get("GUNICORN_BIND", "0.0.0.0:8000")

# ============ LOGGING & MONITORING ============

# Access log format
access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'
)

# Log file locations
accesslog = os.environ.get("GUNICORN_ACCESS_LOG", "-")  # stdout
errorlog = os.environ.get("GUNICORN_ERROR_LOG", "-")    # stderr

# Logging level
loglevel = os.environ.get("GUNICORN_LOG_LEVEL", "info")

# ============ PROCESS NAMING ============

proc_name = "hemophilia-clinic-api"

# ============ SERVER MECHANICS ============

# Preload app code in master
preload_app = True

# Graceful timeout for worker shutdown
graceful_timeout = 30

# ============ SERVER HOOKS ============

def on_starting(server):
    """Called just before the master process is initialized"""
    print(f"[{datetime.now()}] Starting Gunicorn server...")
    print(f"Workers: {workers}")
    print(f"Worker class: {worker_class}")
    print(f"Threads per worker: {threads}")


def when_ready(server):
    """Called just after the server is started"""
    print(f"[{datetime.now()}] Gunicorn server is ready. Spawning workers")


def on_exit(server):
    """Called just before exiting Gunicorn"""
    print(f"[{datetime.now()}] Shutting down Gunicorn server...")


def worker_int(worker):
    """Called when a worker receives SIGINT"""
    print(f"[{datetime.now()}] Worker {worker.pid} received SIGINT")


def worker_abort(worker):
    """Called when a worker receives SIGABRT"""
    print(f"[{datetime.now()}] Worker {worker.pid} aborted")


# ============ DEPLOYMENT GUIDE ============

"""
DEPLOYMENT INSTRUCTIONS:

1. Install Gunicorn and dependencies:
   pip install gunicorn uvicorn uvloop httptools

2. Run with this config:
   gunicorn -c gunicorn_config.py api_optimized:app

3. With environment variables:
   GUNICORN_WORKERS=8 GUNICORN_BIND=0.0.0.0:8000 gunicorn -c gunicorn_config.py api_optimized:app

4. Behind Nginx (recommended):
   
   # nginx.conf
   upstream gunicorn {
       server 127.0.0.1:8000;
       server 127.0.0.1:8001;
       keepalive 32;
   }
   
   server {
       listen 80;
       client_max_body_size 100M;
       
       location / {
           proxy_pass http://gunicorn;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
           proxy_redirect off;
       }
   }

5. With systemd service:
   
   # /etc/systemd/system/hemophilia-clinic.service
   [Unit]
   Description=Hemophilia Clinic API
   After=network.target
   
   [Service]
   User=www-data
   WorkingDirectory=/app
   ExecStart=/app/.venv/bin/gunicorn \\
       -c gunicorn_config.py \\
       api_optimized:app
   Restart=always
   RestartSec=5
   
   [Install]
   WantedBy=multi-user.target
   
   # Enable and start:
   systemctl enable hemophilia-clinic
   systemctl start hemophilia-clinic

6. Performance monitoring:
   # Watch Gunicorn stats
   watch -n 1 'ps aux | grep gunicorn'
   
   # Monitor with systemd
   journalctl -u hemophilia-clinic -f

7. Load balancing with multiple instances:
   # Run multiple Gunicorn instances on different ports
   GUNICORN_BIND=0.0.0.0:8000 gunicorn -c gunicorn_config.py api_optimized:app &
   GUNICORN_BIND=0.0.0.0:8001 gunicorn -c gunicorn_config.py api_optimized:app &
   
   # Then Nginx load balances between them

8. Docker deployment:
   
   # Dockerfile
   FROM python:3.12-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["gunicorn", "-c", "gunicorn_config.py", "api_optimized:app"]
   
   # docker-compose.yml
   version: '3.8'
   services:
     api:
       build: .
       ports:
         - "8000:8000"
       environment:
         GUNICORN_WORKERS: 8
         GUNICORN_LOG_LEVEL: info
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
         interval: 30s
         timeout: 10s
         retries: 3
     
     nginx:
       image: nginx:latest
       ports:
         - "80:80"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf
       depends_on:
         - api

9. Performance benchmarking:
   # Use Apache Bench
   ab -n 1000 -c 100 http://localhost:8000/health
   
   # Use wrk
   wrk -t4 -c100 -d30s http://localhost:8000/health

10. Monitoring best practices:
    - Monitor CPU utilization (target: 50-70%)
    - Monitor memory per worker (target: <150MB)
    - Monitor request latency (target: <200ms)
    - Monitor error rate (target: <0.1%)
    - Check queue size regularly
    - Monitor database connections
    - Set up alerts for resource exhaustion
"""
