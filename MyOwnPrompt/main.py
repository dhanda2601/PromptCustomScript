#!/usr/bin/env python3
"""
FastAPI AI Prompt Generator Service
A REST API service for generating optimized prompts for software developers
"""

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import random
from datetime import datetime
from enum import Enum
import uvicorn

# Pydantic Models
class DomainType(str, Enum):
    ECOMMERCE = "ecommerce"
    BANKING = "banking"
    GENERAL = "general"

class TechnologyType(str, Enum):
    PYTHON = "python"
    NODEJS = "nodejs"
    DATABASE = "database"
    AWS = "aws"

class PromptRequest(BaseModel):
    domain: DomainType
    task_description: str = Field(..., description="Description of the task/problem")
    technologies: Optional[List[TechnologyType]] = Field(default=None, description="Required technologies")
    requirements: Optional[List[str]] = Field(default=None, description="Specific functional requirements")
    output_type: Optional[str] = Field(default=None, description="Desired output format")
    include_examples: bool = Field(default=True, description="Include practical examples")

class CodeReviewRequest(BaseModel):
    code_snippet: str = Field(..., description="Code to review")
    focus_areas: Optional[List[str]] = Field(default=None, description="Areas to focus on")

class DebuggingRequest(BaseModel):
    error_description: str = Field(..., description="Description of the error/issue")
    code_context: Optional[str] = Field(default=None, description="Relevant code context")

class ArchitectureRequest(BaseModel):
    system_description: str = Field(..., description="System to design")
    scale_requirements: Optional[str] = Field(default=None, description="Scale and performance requirements")

class PromptResponse(BaseModel):
    prompt: str
    generated_at: datetime
    prompt_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

# FastAPI App
app = FastAPI(
    title="AI Prompt Generator for Developers",
    description="Generate optimized prompts for software development tasks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

class PromptGeneratorService:
    
    def __init__(self):
        self.context_templates = {
            "ecommerce": [
                "I'm building an e-commerce platform that handles",
                "Working on a retail system that needs to",
                "Developing a marketplace application for",
                "Creating an online store backend that"
            ],
            "banking": [
                "I'm developing a banking application that requires",
                "Building a financial system that needs to",
                "Working on a payment processing system for",
                "Creating a fintech solution that handles"
            ],
            "general": [
                "I'm a senior developer working on a project that",
                "Building a scalable system that needs to",
                "Developing an application that requires",
                "Working on a solution that must"
            ]
        }
        
        self.tech_constraints = {
            "python": ["Python 3.9+", "Django/Flask", "FastAPI", "asyncio", "SQLAlchemy"],
            "nodejs": ["Node.js 18+", "Express.js", "TypeScript", "async/await", "Prisma/Mongoose"],
            "database": ["PostgreSQL", "MySQL", "Redis", "MongoDB", "transaction handling"],
            "aws": ["Lambda", "RDS", "S3", "API Gateway", "CloudWatch", "IAM"]
        }
        
        self.output_formats = [
            "working code with comprehensive comments",
            "production-ready solution with error handling",
            "detailed implementation with unit tests",
            "step-by-step solution with explanations",
            "optimized code following best practices"
        ]
    
    def generate_context_prompt(self, domain: str, task_description: str) -> str:
        """Generate context-rich prompt opening"""
        templates = self.context_templates.get(domain, self.context_templates["general"])
        context_start = random.choice(templates)
        
        experience_context = (
            "As a software developer with 10+ years of experience in e-commerce, "
            "banking, and enterprise applications, "
        )
        
        return f"{experience_context}{context_start} {task_description}."
    
    def add_technical_constraints(self, technologies: List[str]) -> str:
        """Add technical constraints to prompt"""
        constraints = []
        for tech in technologies:
            if tech in self.tech_constraints:
                tech_items = self.tech_constraints[tech]
                constraints.extend(random.sample(tech_items, min(2, len(tech_items))))
        
        if constraints:
            return f"\n\nTechnical Requirements:\n- Must use {', '.join(constraints[:3])}\n- Should follow industry best practices\n- Needs proper error handling and logging"
        return ""
    
    def add_specific_requirements(self, requirements: List[str]) -> str:
        """Add specific functional requirements"""
        if not requirements:
            return ""
        
        req_text = "\n\nSpecific Requirements:\n"
        for i, req in enumerate(requirements, 1):
            req_text += f"{i}. {req}\n"
        
        return req_text
    
    def add_output_specification(self, output_type: str = None) -> str:
        """Specify desired output format"""
        if not output_type:
            output_type = random.choice(self.output_formats)
        
        return f"\n\nPlease provide {output_type}."
    
    def generate_complete_prompt(self, 
                                domain: str,
                                task_description: str,
                                technologies: List[str] = None,
                                requirements: List[str] = None,
                                output_type: str = None,
                                include_examples: bool = True) -> str:
        """Generate a complete, optimized prompt"""
        
        # Start with context
        prompt = self.generate_context_prompt(domain, task_description)
        
        # Add technical constraints
        if technologies:
            prompt += self.add_technical_constraints(technologies)
        
        # Add specific requirements
        if requirements:
            prompt += self.add_specific_requirements(requirements)
        
        # Add example request if needed
        if include_examples:
            prompt += "\n\nPlease include practical examples and explain your approach."
        
        # Add output specification
        prompt += self.add_output_specification(output_type)
        
        return prompt
    
    def generate_code_review_prompt(self, code_snippet: str, focus_areas: List[str] = None) -> str:
        """Generate prompt for code review"""
        base_prompt = (
            "As an experienced developer, please review this code snippet. "
            "I'm particularly interested in performance, security, and maintainability."
        )
        
        if focus_areas:
            focus_text = f"\n\nPlease focus on: {', '.join(focus_areas)}"
            base_prompt += focus_text
        
        base_prompt += f"\n\nCode to review:\n```python\n{code_snippet}\n```"
        base_prompt += "\n\nProvide specific suggestions for improvement with explanations."
        
        return base_prompt
    
    def generate_debugging_prompt(self, error_description: str, code_context: str = None) -> str:
        """Generate prompt for debugging help"""
        prompt = (
            "I'm encountering an issue in my production system. "
            f"Here's the problem: {error_description}"
        )
        
        if code_context:
            prompt += f"\n\nRelevant code context:\n```python\n{code_context}\n```"
        
        prompt += (
            "\n\nPlease help me:\n"
            "1. Identify the root cause\n"
            "2. Provide a solution\n"
            "3. Suggest preventive measures\n"
            "4. Recommend testing strategies"
        )
        
        return prompt
    
    def generate_architecture_prompt(self, system_description: str, scale_requirements: str = None) -> str:
        """Generate prompt for system architecture design"""
        prompt = (
            f"I need to design the architecture for {system_description}. "
            "Given my experience with e-commerce and banking systems, "
            "I'm looking for a scalable, secure, and maintainable solution."
        )
        
        if scale_requirements:
            prompt += f"\n\nScale requirements: {scale_requirements}"
        
        prompt += (
            "\n\nPlease provide:\n"
            "1. High-level architecture diagram description\n"
            "2. Technology stack recommendations\n"
            "3. Database design considerations\n"
            "4. Security implementation strategies\n"
            "5. Deployment and monitoring approach"
        )
        
        return prompt

# Initialize service
prompt_service = PromptGeneratorService()

# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Prompt Generator API for Developers",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "generate_prompt": "/generate-prompt",
            "code_review": "/code-review",
            "debugging": "/debugging",
            "architecture": "/architecture",
            "health": "/health"
        }
    }

@app.post("/generate-prompt", response_model=PromptResponse)
async def generate_prompt(request: PromptRequest):
    """Generate a complete development prompt"""
    try:
        prompt = prompt_service.generate_complete_prompt(
            domain=request.domain.value,
            task_description=request.task_description,
            technologies=[tech.value for tech in request.technologies] if request.technologies else None,
            requirements=request.requirements,
            output_type=request.output_type,
            include_examples=request.include_examples
        )
        
        return PromptResponse(
            prompt=prompt,
            generated_at=datetime.now(),
            prompt_type="development",
            metadata={
                "domain": request.domain.value,
                "technologies": [tech.value for tech in request.technologies] if request.technologies else [],
                "requirements_count": len(request.requirements) if request.requirements else 0
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating prompt: {str(e)}")

@app.post("/code-review", response_model=PromptResponse)
async def generate_code_review_prompt(request: CodeReviewRequest):
    """Generate a code review prompt"""
    try:
        prompt = prompt_service.generate_code_review_prompt(
            code_snippet=request.code_snippet,
            focus_areas=request.focus_areas
        )
        
        return PromptResponse(
            prompt=prompt,
            generated_at=datetime.now(),
            prompt_type="code_review",
            metadata={
                "code_length": len(request.code_snippet),
                "focus_areas": request.focus_areas or []
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating code review prompt: {str(e)}")

@app.post("/debugging", response_model=PromptResponse)
async def generate_debugging_prompt(request: DebuggingRequest):
    """Generate a debugging prompt"""
    try:
        prompt = prompt_service.generate_debugging_prompt(
            error_description=request.error_description,
            code_context=request.code_context
        )
        
        return PromptResponse(
            prompt=prompt,
            generated_at=datetime.now(),
            prompt_type="debugging",
            metadata={
                "has_code_context": bool(request.code_context),
                "error_description_length": len(request.error_description)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating debugging prompt: {str(e)}")

@app.post("/architecture", response_model=PromptResponse)
async def generate_architecture_prompt(request: ArchitectureRequest):
    """Generate an architecture design prompt"""
    try:
        prompt = prompt_service.generate_architecture_prompt(
            system_description=request.system_description,
            scale_requirements=request.scale_requirements
        )
        
        return PromptResponse(
            prompt=prompt,
            generated_at=datetime.now(),
            prompt_type="architecture",
            metadata={
                "has_scale_requirements": bool(request.scale_requirements),
                "system_description_length": len(request.system_description)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating architecture prompt: {str(e)}")

@app.get("/templates", response_model=dict)
async def get_prompt_templates():
    """Get example prompt templates"""
    examples = {
        "ecommerce_example": {
            "domain": "ecommerce",
            "task_description": "build a shopping cart system with inventory management",
            "technologies": ["python", "database", "aws"],
            "requirements": [
                "Handle concurrent users",
                "Real-time inventory updates",
                "Integration with payment gateways"
            ]
        },
        "banking_example": {
            "domain": "banking",
            "task_description": "create a transaction processing system with audit trails",
            "technologies": ["python", "database"],
            "requirements": [
                "ACID compliance",
                "Audit logging",
                "Risk assessment"
            ]
        },
        "code_review_example": {
            "code_snippet": "def process_payment(amount, card_number):\n    if amount > 0:\n        return {'status': 'success'}\n    return {'status': 'failed'}",
            "focus_areas": ["security vulnerabilities", "error handling", "input validation"]
        }
    }
    
    return {"templates": examples, "usage": "Use these as examples for your API requests"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "service": "prompt-generator-api"
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

# Run the server
if __name__ == "__main__":
    print("Starting FastAPI Prompt Generator Service...")
    print("API Documentation: http://localhost:8000/docs")
    print("API Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )