"""
main.py - FastAPI Application for QuickBasket AI
Complete Backend with Authentication, Database ORM, ML/NLP Intelligence,
Shopping Cart, Checkout Orders, and Static Frontend SPA Serving.
"""

import os
import sys
import json
import uuid
from datetime import datetime
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.database import engine, get_db, init_db, User, CartItem, Order, SavedRecipe, Review
from backend.auth import hash_password, verify_password, create_access_token, get_current_user, get_optional_user
from backend.schemas import (
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    TokenResponse,
    ProductResponse,
    CartSyncRequest,
    CheckoutRequest,
    OrderResponse,
    ReviewCreateRequest,
    ReviewResponse,
    RecommendRequest,
    RecipeCopilotRequest,
    RecipeSaveRequest,
    ColdStartRequest
)
from backend.ml_service import MLService

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Initializing database tables & seeding demo user...")
    init_db()
    print("🧠 Warming up ML & NLP models...")
    MLService.get_instance()
    print("✨ QuickBasket AI is ready!")
    yield

app = FastAPI(
    title="QuickBasket AI - Market Basket & NLP Intelligence API",
    description="Full-Stack Quick-Commerce Platform fusing FP-Growth Association Mining with NLP Semantic Intelligence.",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# AUTH ENDPOINTS
# ==========================================
@app.post("/api/auth/register", response_model=TokenResponse)
def register(req: UserRegisterRequest, db: Session = Depends(get_db)):
    # Check email exists
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=400, detail="Email is already registered.")
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(status_code=400, detail="Username is already taken.")

    new_user = User(
        email=req.email,
        username=req.username,
        hashed_password=hash_password(req.password),
        full_name=req.full_name,
        loyalty_points=150
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token({"sub": new_user.username})
    return {"access_token": token, "token_type": "bearer", "user": new_user}


@app.post("/api/auth/login", response_model=TokenResponse)
def login(req: UserLoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(
        (User.username == req.username_or_email) | (User.email == req.username_or_email)
    ).first()

    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password"
        )

    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer", "user": user}


@app.get("/api/auth/me", response_model=UserResponse)
def get_profile(current_user: User = Depends(get_current_user)):
    return current_user


# ==========================================
# CATALOG ENDPOINTS
# ==========================================
@app.get("/api/catalog/products")
def get_products(aisle: Optional[str] = None, search: Optional[str] = None):
    ml = MLService.get_instance()
    return ml.get_all_products(aisle=aisle, search=search)


@app.get("/api/catalog/products/{product_id}")
def get_product(product_id: str):
    ml = MLService.get_instance()
    prod = ml.get_product_detail(product_id)
    if not prod:
        raise HTTPException(status_code=404, detail="Product not found.")
    return prod


@app.get("/api/catalog/aisles")
def get_aisles():
    ml = MLService.get_instance()
    return ml.get_aisles_summary()


# ==========================================
# ML & NLP INTELLIGENCE ENDPOINTS
# ==========================================
@app.post("/api/ml/recommend")
def get_recommendations(req: RecommendRequest):
    ml = MLService.get_instance()
    return ml.recommend_for_cart(
        cart_ids=req.cart_items,
        top_n=req.top_n,
        filter_substitutes=req.filter_substitutes
    )


@app.post("/api/ml/recipe-copilot")
def recipe_copilot(req: RecipeCopilotRequest):
    ml = MLService.get_instance()
    return ml.synthesize_recipe_copilot(
        prompt=req.prompt,
        servings=req.servings,
        max_budget=req.max_budget or 35.0
    )


@app.get("/api/ml/bundles")
def get_bundles():
    ml = MLService.get_instance()
    return ml.bundles


@app.get("/api/ml/semantic-map")
def get_semantic_map(cart: Optional[str] = None):
    ml = MLService.get_instance()
    cart_ids = [c.strip() for c in cart.split(",") if c.strip()] if cart else []
    return ml.get_semantic_2d_map(cart_ids=cart_ids)


@app.post("/api/ml/cart-diagnostic")
def get_cart_diagnostic(req: RecommendRequest):
    ml = MLService.get_instance()
    return ml.evaluate_cart_diagnostics(cart_ids=req.cart_items)


@app.post("/api/ml/cold-start")
def solve_cold_start(req: ColdStartRequest):
    ml = MLService.get_instance()
    return ml.solve_cold_start(
        name=req.name,
        description=req.description,
        ingredients=req.ingredients,
        aisle=req.aisle
    )


@app.get("/api/ml/benchmarks")
def get_benchmarks():
    ml = MLService.get_instance()
    return ml.get_benchmarks()


# ==========================================
# CART & PERSISTENCE ENDPOINTS
# ==========================================
@app.get("/api/cart")
def get_cart(current_user: Optional[User] = Depends(get_optional_user), db: Session = Depends(get_db)):
    if not current_user:
        return {"items": []}
    items = db.query(CartItem).filter(CartItem.user_id == current_user.id).all()
    return {"items": [{"product_id": itm.product_id, "quantity": itm.quantity} for itm in items]}


@app.post("/api/cart/sync")
def sync_cart(req: CartSyncRequest, current_user: Optional[User] = Depends(get_optional_user), db: Session = Depends(get_db)):
    if not current_user:
        return {"status": "ok", "synced": False, "note": "Guest cart session"}

    # Clear and replace user's cart items
    db.query(CartItem).filter(CartItem.user_id == current_user.id).delete()
    for itm in req.items:
        db.add(CartItem(
            user_id=current_user.id,
            product_id=itm.product_id,
            quantity=itm.quantity
        ))
    db.commit()
    return {"status": "ok", "synced": True, "count": len(req.items)}


@app.post("/api/cart/checkout", response_model=OrderResponse)
def checkout(
    req: CheckoutRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    cart_items = db.query(CartItem).filter(CartItem.user_id == current_user.id).all()
    if not cart_items:
        raise HTTPException(status_code=400, detail="Shopping cart is empty.")

    ml = MLService.get_instance()
    order_items = []
    subtotal = 0.0
    has_cold = False

    for c in cart_items:
        prod = next((p for p in ml.catalog if p["id"] == c.product_id), None)
        if prod:
            price = prod["price"]
            subtotal += price * c.quantity
            if prod["cold_chain"]:
                has_cold = True
            order_items.append({
                "product_id": prod["id"],
                "name": prod["name"],
                "price": price,
                "quantity": c.quantity,
                "image_url": prod["image_url"]
            })

    # Free delivery if subtotal >= $15.00
    delivery_fee = 0.0 if subtotal >= 15.00 else 1.99
    discount = round(subtotal * 0.10, 2) if req.promo_code and req.promo_code.upper() == "AI10" else 0.0
    total = round(subtotal - discount + delivery_fee, 2)

    order_id = f"ORD-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    new_order = Order(
        id=order_id,
        user_id=current_user.id,
        items_json=json.dumps(order_items),
        subtotal=round(subtotal, 2),
        discount=discount,
        delivery_fee=delivery_fee,
        total_amount=total,
        status="Confirmed (10-Min Fast Track)",
        cold_chain=has_cold,
        delivery_address=req.delivery_address,
        payment_method=req.payment_method,
        estimated_arrival_minutes=10
    )
    db.add(new_order)

    # Award loyalty points (10 points per dollar spent)
    current_user.loyalty_points += int(total * 10)

    # Empty user's cart
    db.query(CartItem).filter(CartItem.user_id == current_user.id).delete()
    db.commit()
    db.refresh(new_order)

    return {
        "id": new_order.id,
        "user_id": new_order.user_id,
        "items": order_items,
        "subtotal": new_order.subtotal,
        "discount": new_order.discount,
        "delivery_fee": new_order.delivery_fee,
        "total_amount": new_order.total_amount,
        "status": new_order.status,
        "cold_chain": new_order.cold_chain,
        "delivery_address": new_order.delivery_address,
        "payment_method": new_order.payment_method,
        "estimated_arrival_minutes": new_order.estimated_arrival_minutes,
        "created_at": new_order.created_at
    }


@app.get("/api/orders", response_model=List[OrderResponse])
def get_orders(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    orders = db.query(Order).filter(Order.user_id == current_user.id).order_by(Order.created_at.desc()).all()
    results = []
    for o in orders:
        results.append({
            "id": o.id,
            "user_id": o.user_id,
            "items": json.loads(o.items_json),
            "subtotal": o.subtotal,
            "discount": o.discount,
            "delivery_fee": o.delivery_fee,
            "total_amount": o.total_amount,
            "status": o.status,
            "cold_chain": o.cold_chain,
            "delivery_address": o.delivery_address,
            "payment_method": o.payment_method,
            "estimated_arrival_minutes": o.estimated_arrival_minutes,
            "created_at": o.created_at
        })
    return results


# ==========================================
# REVIEWS & SENTIMENT ENDPOINTS
# ==========================================
@app.get("/api/reviews/{product_id}")
def get_product_reviews(product_id: str, db: Session = Depends(get_db)):
    db_reviews = db.query(Review).filter(Review.product_id == product_id).order_by(Review.created_at.desc()).limit(10).all()
    if db_reviews:
        return [
            {
                "id": r.id,
                "product_id": r.product_id,
                "author_name": r.author_name,
                "rating": r.rating,
                "review_text": r.review_text,
                "sentiment_score": r.sentiment_score,
                "sentiment_label": r.sentiment_label,
                "created_at": r.created_at
            }
            for r in db_reviews
        ]

    # Fallback to reviews from NLP engine sample
    ml = MLService.get_instance()
    pairings = ml.nlp.pairing_insights.get(product_id, [])
    sample = [
        {
            "id": 1,
            "product_id": product_id,
            "author_name": "Verified Foodie",
            "rating": 5,
            "review_text": "Exceptional quality and freshness. Paired wonderfully with complementary ingredients!",
            "sentiment_score": 0.88,
            "sentiment_label": "POSITIVE",
            "created_at": datetime.utcnow()
        }
    ]
    return sample


@app.post("/api/reviews", response_model=ReviewResponse)
def submit_review(
    req: ReviewCreateRequest,
    current_user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    # Compute real-time VADER sentiment
    from src.nlp_engine import VADER
    vs = VADER.polarity_scores(req.review_text)
    compound = vs["compound"]
    label = "POSITIVE" if compound >= 0.05 else ("NEGATIVE" if compound <= -0.05 else "NEUTRAL")

    author = req.author_name or (current_user.full_name if current_user else "Quick Shopper")
    user_id = current_user.id if current_user else None

    rev = Review(
        user_id=user_id,
        product_id=req.product_id,
        author_name=author,
        rating=req.rating,
        review_text=req.review_text,
        sentiment_score=compound,
        sentiment_label=label
    )
    db.add(rev)
    db.commit()
    db.refresh(rev)
    return rev


# ==========================================
# SAVED RECIPES ENDPOINTS
# ==========================================
@app.get("/api/recipes/saved")
def get_saved_recipes(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    recipes = db.query(SavedRecipe).filter(SavedRecipe.user_id == current_user.id).all()
    return [
        {
            "id": r.id,
            "title": r.title,
            "prompt": r.prompt,
            "cuisine": r.cuisine,
            "recipe_json": json.loads(r.recipe_json),
            "created_at": r.created_at
        }
        for r in recipes
    ]


@app.post("/api/recipes/save")
def save_recipe(
    req: RecipeSaveRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    new_rec = SavedRecipe(
        user_id=current_user.id,
        title=req.title,
        prompt=req.prompt,
        cuisine=req.cuisine,
        recipe_json=json.dumps(req.recipe_json)
    )
    db.add(new_rec)
    db.commit()
    db.refresh(new_rec)
    return {"status": "saved", "id": new_rec.id}


# ==========================================
# STATIC FRONTEND SERVING
# ==========================================
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    @app.get("/{catchall:path}")
    def serve_frontend_routes(catchall: str):
        file_path = os.path.join(FRONTEND_DIR, catchall)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
