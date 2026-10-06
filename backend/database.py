"""
database.py - SQLAlchemy Database Models & Engine for QuickBasket AI
Manages persistent users, shopping carts, checkout orders, reviews, and saved recipes.
"""

import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "ecommerce.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    avatar_url = Column(String, default="https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150")
    loyalty_points = Column(Integer, default=250)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    cart_items = relationship("CartItem", back_populates="user", cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="user", cascade="all, delete-orphan")
    saved_recipes = relationship("SavedRecipe", back_populates="user", cascade="all, delete-orphan")


class CartItem(Base):
    __tablename__ = "cart_items"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    product_id = Column(String, nullable=False)
    quantity = Column(Integer, default=1)
    added_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="cart_items")


class Order(Base):
    __tablename__ = "orders"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    items_json = Column(Text, nullable=False)  # JSON-encoded array of cart items
    subtotal = Column(Float, nullable=False)
    discount = Column(Float, default=0.0)
    delivery_fee = Column(Float, default=0.0)
    total_amount = Column(Float, nullable=False)
    status = Column(String, default="Confirmed")  # Confirmed, Packing, Out for Delivery, Delivered
    cold_chain = Column(Boolean, default=False)
    delivery_address = Column(String, default="742 Evergreen Terrace, San Francisco, CA")
    payment_method = Column(String, default="Instant QuickPay (Apple Pay)")
    estimated_arrival_minutes = Column(Integer, default=10)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="orders")


class SavedRecipe(Base):
    __tablename__ = "saved_recipes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    prompt = Column(String, nullable=False)
    cuisine = Column(String, default="Italian")
    recipe_json = Column(Text, nullable=False)  # ingredients, steps, estimated_cost
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="saved_recipes")


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    product_id = Column(String, index=True, nullable=False)
    author_name = Column(String, nullable=False)
    rating = Column(Integer, nullable=False)
    review_text = Column(Text, nullable=False)
    sentiment_score = Column(Float, default=0.0)
    sentiment_label = Column(String, default="POSITIVE")
    created_at = Column(DateTime, default=datetime.utcnow)


def get_db():
    """Dependency for acquiring database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initializes tables and seeds default demo data if missing."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        from backend.auth import hash_password
        demo_email = "demo@antigravity.ai"
        user = db.query(User).filter(User.email == demo_email).first()
        if not user:
            print("🌱 Seeding default demo user (demo@antigravity.ai)...")
            demo_user = User(
                email=demo_email,
                username="chef_alex",
                hashed_password=hash_password("Password123!"),
                full_name="Chef Alex Rivera",
                avatar_url="https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150",
                loyalty_points=500,
                is_admin=True
            )
            db.add(demo_user)
            db.commit()
            db.refresh(demo_user)

            sample_items = [
                {"product_id": "PROD_001", "name": "Barilla Penne Rigate Pasta", "price": 2.49, "quantity": 1, "image_url": "https://images.unsplash.com/photo-1551462147-37885acc36f1?w=500"},
                {"product_id": "PROD_004", "name": "Rao's Homemade Marinara Sauce", "price": 8.99, "quantity": 1, "image_url": "https://images.unsplash.com/photo-1572449043416-55f4685c9bb7?w=500"},
                {"product_id": "PROD_011", "name": "Sundried Tomatoes in Herb Oil", "price": 4.99, "quantity": 1, "image_url": "https://images.unsplash.com/photo-1567306226416-28f0efdc88ce?w=500"}
            ]
            sample_order = Order(
                id="ORD-202610-8491",
                user_id=demo_user.id,
                items_json=json.dumps(sample_items),
                subtotal=16.47,
                discount=1.50,
                delivery_fee=0.0,
                total_amount=14.97,
                status="Delivered",
                cold_chain=False,
                delivery_address="742 Evergreen Terrace, San Francisco, CA",
                payment_method="Instant QuickPay (Apple Pay)",
                estimated_arrival_minutes=8
            )
            db.add(sample_order)

            saved_rec = SavedRecipe(
                user_id=demo_user.id,
                title="Rustic Artisanal Marinara Penne",
                prompt="Authentic creamy Italian pasta dinner for date night",
                cuisine="Italian",
                recipe_json=json.dumps({
                    "prep_time": "15 Mins",
                    "difficulty": "Easy",
                    "estimated_cost": "$16.47",
                    "steps": [
                        "Boil salted water in large pot; cook Penne Rigate until al dente (10 mins).",
                        "Simmer Rao's Homemade Marinara Sauce in saucepan with crushed garlic.",
                        "Drain pasta, toss directly into hot marinara sauce.",
                        "Garnish with freshly torn sundried tomatoes and grated parmesan cheese."
                    ]
                })
            )
            db.add(saved_rec)
            db.commit()
            print("✅ Default demo user and sample order seeded successfully.")
    finally:
        db.close()
