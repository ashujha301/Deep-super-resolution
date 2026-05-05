from fastapi import APIRouter

from backend.api.routes import health, platforms, optimization

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router)
api_router.include_router(platforms.router)
api_router.include_router(optimization.router)