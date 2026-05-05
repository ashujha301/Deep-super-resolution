from fastapi import APIRouter

from backend.inference.platform_optimizer import PLATFORM_RULES

router = APIRouter(prefix="/platforms", tags=["Platforms"])


@router.get("")
def get_platforms():
    return {
        "platforms": list(PLATFORM_RULES.keys()),
        "rules": PLATFORM_RULES,
    }