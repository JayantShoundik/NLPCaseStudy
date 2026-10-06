"""
schemas.py - Pydantic Request & Response Models for QuickBasket AI
Defines contracts for auth, shopping cart, orders, reviews, and ML/NLP intelligence endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ==========================================
# AUTH SCHEMAS
# ==========================================
class UserRegisterRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=100)
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    full_name: str = Field(..., min_length=2)


class UserLoginRequest(BaseModel):
    username_or_email: str
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: str
    avatar_url: Optional[str] = None
    loyalty_points: int
    is_admin: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ==========================================
# PRODUCT & CATALOG SCHEMAS
# ==========================================
class ProductResponse(BaseModel):
    id: str
    name: str
    aisle: str
    price: float
    description: str
    ingredients: str
    tags: str
    image_url: str
    pack_spec: str
    calories: str
    cold_chain: bool
    avg_rating: Optional[float] = 4.8
    review_count: Optional[int] = 120


class AisleInfo(BaseModel):
    name: str
    icon: str
    count: int
    color: str


# ==========================================
# CART & CHECKOUT SCHEMAS
# ==========================================
class CartItemSchema(BaseModel):
    product_id: str
    quantity: int = 1


class CartSyncRequest(BaseModel):
    items: List[CartItemSchema]


class CheckoutRequest(BaseModel):
    delivery_address: str = "742 Evergreen Terrace, San Francisco, CA"
    payment_method: str = "Instant QuickPay (Apple Pay)"
    promo_code: Optional[str] = None


class OrderResponse(BaseModel):
    id: str
    user_id: int
    items: List[Dict[str, Any]]
    subtotal: float
    discount: float
    delivery_fee: float
    total_amount: float
    status: str
    cold_chain: bool
    delivery_address: str
    payment_method: str
    estimated_arrival_minutes: int
    created_at: datetime


# ==========================================
# REVIEW SCHEMAS
# ==========================================
class ReviewCreateRequest(BaseModel):
    product_id: str
    rating: int = Field(..., ge=1, le=5)
    review_text: str = Field(..., min_length=5)
    author_name: Optional[str] = None


class ReviewResponse(BaseModel):
    id: int
    product_id: str
    author_name: str
    rating: int
    review_text: str
    sentiment_score: float
    sentiment_label: str
    created_at: datetime

    class Config:
        from_attributes = True


# ==========================================
# ML & NLP INTELLIGENCE SCHEMAS
# ==========================================
class RecommendRequest(BaseModel):
    cart_items: List[str]
    top_n: int = 5
    filter_substitutes: bool = True


class RecipeCopilotRequest(BaseModel):
    prompt: str = Field(..., min_length=3)
    servings: int = 2
    max_budget: Optional[float] = 25.0


class RecipeSaveRequest(BaseModel):
    title: str
    prompt: str
    cuisine: str
    recipe_json: Dict[str, Any]


class ColdStartRequest(BaseModel):
    name: str
    description: str
    ingredients: str
    aisle: str
