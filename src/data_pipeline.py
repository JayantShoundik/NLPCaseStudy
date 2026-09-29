"""
data_pipeline.py - High-Throughput Supermarket Data Generation & Preprocessing Engine
Generates realistic product catalogs, transactions (with realistic co-purchases), and customer reviews.
"""

import os
import random
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

PRODUCT_CATALOG = [
    # 1. Pasta & Italian Pantry
    {"id": "PROD_001", "name": "Italian Penne Rigate Pasta", "category": "Pasta & Italian", "price": 2.49,
     "description": "Authentic bronze-cut Italian penne rigate pasta made from 100% durum wheat semolina. Perfect ridges hold heavy marinara and creamy sauces.",
     "ingredients": "Durum wheat semolina, water",
     "pairing_keywords": ["marinara sauce", "parmesan cheese", "pesto", "garlic bread", "olive oil"]},
    {"id": "PROD_002", "name": "Organic Spaghetti Pasta", "category": "Pasta & Italian", "price": 2.29,
     "description": "Certified organic classic spaghetti pasta with optimal al dente texture. Best served with rich bolognese, tomato basil sauce, or simple garlic and oil.",
     "ingredients": "Organic durum wheat semolina, water",
     "pairing_keywords": ["tomato sauce", "parmesan cheese", "garlic bread", "meatballs", "olive oil"]},
    {"id": "PROD_003", "name": "Fettuccine Egg Pasta", "category": "Pasta & Italian", "price": 3.19,
     "description": "Traditional wide ribbon egg fettuccine pasta with silky texture. Rich and satisfying when tossed with alfredo sauce, butter, and freshly grated parmesan.",
     "ingredients": "Durum wheat semolina, fresh pasteurized eggs, water",
     "pairing_keywords": ["alfredo sauce", "parmesan cheese", "garlic bread", "butter", "cracked black pepper"]},
    {"id": "PROD_004", "name": "Organic Basil Marinara Sauce", "category": "Pasta & Italian", "price": 4.99,
     "description": "Slow-simmered slow-cooked Italian marinara sauce with San Marzano tomatoes, fresh sweet basil, garlic, and extra virgin olive oil.",
     "ingredients": "San Marzano tomatoes, extra virgin olive oil, fresh basil, garlic, sea salt, oregano",
     "pairing_keywords": ["penne pasta", "spaghetti", "parmesan cheese", "garlic bread", "mozzarella"]},
    {"id": "PROD_005", "name": "Creamy Garlic Alfredo Sauce", "category": "Pasta & Italian", "price": 4.79,
     "description": "Velvety homestyle alfredo sauce prepared with real heavy cream, aged parmesan cheese, and roasted minced garlic.",
     "ingredients": "Heavy cream, parmesan cheese, butter, roasted garlic, black pepper, sea salt",
     "pairing_keywords": ["fettuccine", "parmesan cheese", "garlic bread", "chicken breast", "black pepper"]},
    {"id": "PROD_006", "name": "Traditional Basil Pesto Genovese", "category": "Pasta & Italian", "price": 5.49,
     "description": "Aromatic Genovese pesto crafted from fresh crushed basil leaves, pine nuts, garlic, extra virgin olive oil, and aged Pecorino Romano.",
     "ingredients": "Fresh basil, extra virgin olive oil, pine nuts, Pecorino Romano, sea salt, garlic",
     "pairing_keywords": ["penne pasta", "spaghetti", "cherry tomatoes", "mozzarella", "crusty bread"]},
    {"id": "PROD_007", "name": "Aged Parmigiano Reggiano Wedge", "category": "Pasta & Italian", "price": 7.99,
     "description": "Authentic DOP certified 24-month aged Parmigiano Reggiano from Parma Italy. Nutty, crystalline, umami-packed cheese for grating over hot pasta.",
     "ingredients": "Cultured raw cow milk, salt, rennet",
     "pairing_keywords": ["penne pasta", "spaghetti", "marinara sauce", "alfredo sauce", "olive oil"]},
    {"id": "PROD_008", "name": "Fresh Mozzarella Cheese Ball", "category": "Pasta & Italian", "price": 4.49,
     "description": "Creamy, delicate fresh mozzarella packed in salted water. Melts effortlessly on baked ziti or slices beautifully with ripe tomatoes and fresh basil.",
     "ingredients": "Pasteurized cow milk, vinegar, salt, enzymes",
     "pairing_keywords": ["cherry tomatoes", "fresh basil", "balsamic vinegar", "marinara sauce", "pasta"]},
    {"id": "PROD_009", "name": "Extra Virgin Olive Oil Cold Pressed", "category": "Pasta & Italian", "price": 11.99,
     "description": "First cold-pressed unrefined extra virgin olive oil with fruity floral notes and a peppery finish. Essential for dressing pasta, salads, and dipping.",
     "ingredients": "100% cold-pressed olives",
     "pairing_keywords": ["balsamic vinegar", "garlic bread", "pasta", "salad greens", "tomatoes"]},
    {"id": "PROD_010", "name": "Crusty Garlic & Herb Baguette", "category": "Pasta & Italian", "price": 3.79,
     "description": "Artisan baked French baguette generously slathered with roasted garlic butter, parsley, and sea salt. Crisps golden in the oven in 10 minutes.",
     "ingredients": "Enriched wheat flour, butter, roasted garlic, parsley, yeast, salt",
     "pairing_keywords": ["marinara sauce", "pasta", "spaghetti", "parmesan cheese", "soup"]},
    {"id": "PROD_011", "name": "Sundried Tomatoes in Herb Oil", "category": "Pasta & Italian", "price": 4.29,
     "description": "Tender sun-ripened tomatoes packed in extra virgin olive oil with oregano and garlic. Intensely savory sweet addition to pasta and antipasti.",
     "ingredients": "Sundried tomatoes, sunflower oil, extra virgin olive oil, garlic, oregano, capers",
     "pairing_keywords": ["penne pasta", "pesto", "mozzarella", "olive oil", "salad"]},
    {"id": "PROD_012", "name": "Aged Balsamic Vinegar of Modena", "category": "Pasta & Italian", "price": 8.49,
     "description": "Dense, barrel-aged balsamic vinegar IGP from Modena. Syrupy sweet-tart acidity that drizzled over caprese, fresh strawberries, or roasted vegetables.",
     "ingredients": "Cooked grape must, wine vinegar",
     "pairing_keywords": ["extra virgin olive oil", "fresh mozzarella", "cherry tomatoes", "salad"]},

    # 2. Breakfast & Bakery
    {"id": "PROD_013", "name": "Fluffy Buttermilk Pancake Mix", "category": "Breakfast & Bakery", "price": 3.69,
     "description": "Homestyle complete buttermilk pancake and waffle mix. Just add water or milk for thick, golden, fluffy flapjacks with melt-in-mouth texture.",
     "ingredients": "Enriched bleached flour, buttermilk solids, sugar, baking soda, salt",
     "pairing_keywords": ["pure maple syrup", "unsalted butter", "fresh eggs", "breakfast sausage", "strawberries"]},
    {"id": "PROD_014", "name": "100% Pure Vermont Maple Syrup", "category": "Breakfast & Bakery", "price": 9.99,
     "description": "Grade A amber rich color pure Vermont maple syrup tapped from sugar maple trees. Free of artificial colors, high-fructose corn syrup, and preservatives.",
     "ingredients": "100% pure maple syrup",
     "pairing_keywords": ["pancake mix", "rolled oats", "waffles", "butter", "french toast"]},
    {"id": "PROD_015", "name": "Creamy Unsalted European Butter", "category": "Breakfast & Bakery", "price": 4.99,
     "description": "High-fat 82% European-style unsalted butter churned from fresh sweet cream. Perfect for flaky pastry lamination and melting over hot pancakes.",
     "ingredients": "Pasteurized cream, natural flavoring",
     "pairing_keywords": ["pancake mix", "sourdough bread", "maple syrup", "bagels", "eggs"]},
    {"id": "PROD_016", "name": "Whole Grain Rolled Oats", "category": "Breakfast & Bakery", "price": 3.49,
     "description": "100% whole grain old-fashioned rolled oats. Hearty source of soluble fiber, excellent for morning oatmeal bowls, overnight oats, and baking granola.",
     "ingredients": "Whole grain rolled oats",
     "pairing_keywords": ["pure maple syrup", "almond milk", "berries", "chia seeds", "honey"]},
    {"id": "PROD_017", "name": "Artisan Sourdough Loaf", "category": "Breakfast & Bakery", "price": 4.89,
     "description": "Naturally fermented slow-risen sourdough bread with a blistered crispy crust and airy, chewy crumb with a mild tangy flavor.",
     "ingredients": "Unbleached wheat flour, water, wild sourdough culture, sea salt",
     "pairing_keywords": ["unsalted butter", "avocado", "eggs", "strawberry jam", "cheddar cheese"]},
    {"id": "PROD_018", "name": "All-Butter Flaky Croissants (4-pack)", "category": "Breakfast & Bakery", "price": 5.29,
     "description": "Traditional French bakery style croissants crafted with real butter layers. Flaky golden exterior with honeycomb interior.",
     "ingredients": "Wheat flour, butter, milk, sugar, yeast, egg wash, salt",
     "pairing_keywords": ["strawberry jam", "cappuccino", "butter", "espresso", "eggs"]},
    {"id": "PROD_019", "name": "Organic Strawberry Fruit Spread", "category": "Breakfast & Bakery", "price": 3.99,
     "description": "Naturally sweetened strawberry preserve made with whole ripe organic strawberries and fruit pectin. No high-fructose corn syrup.",
     "ingredients": "Organic strawberries, organic cane sugar, fruit pectin, lemon juice concentrate",
     "pairing_keywords": ["peanut butter", "sourdough bread", "croissants", "pancakes", "waffles"]},
    {"id": "PROD_020", "name": "Plain New York Style Bagels (6-pack)", "category": "Breakfast & Bakery", "price": 3.99,
     "description": "Boiled and baked authentic New York bagels with a chewy golden crust and soft, dense interior. Toast to perfection.",
     "ingredients": "Unbleached wheat flour, water, malted barley flour, sugar, yeast, salt",
     "pairing_keywords": ["cream cheese", "smoked salmon", "butter", "coffee", "eggs"]},
    {"id": "PROD_021", "name": "Original Cream Cheese Spread", "category": "Breakfast & Bakery", "price": 2.99,
     "description": "Smooth and rich spreadable cream cheese made with fresh milk and real cream. Tangy, velvety, and easy to spread over warm bagels.",
     "ingredients": "Pasteurized milk and cream, salt, carob bean gum, cheese culture",
     "pairing_keywords": ["bagels", "smoked salmon", "strawberry jam", "toast", "croissants"]},
    {"id": "PROD_022", "name": "Raw Wildflower Honey", "category": "Breakfast & Bakery", "price": 6.99,
     "description": "Unfiltered raw wildflower honey harvested from natural floral pastures. Rich in enzymes and antioxidants with delicate floral sweetness.",
     "ingredients": "100% pure raw wildflower honey",
     "pairing_keywords": ["green tea", "rolled oats", "greek yogurt", "pancakes", "butter"]},

    # 3. Dairy & Eggs
    {"id": "PROD_023", "name": "Organic Grade A Large Brown Eggs (12-pack)", "category": "Dairy & Eggs", "price": 4.99,
     "description": "Farm-fresh pasture-raised brown eggs with rich golden yolks. Packed with high-quality protein for baking, poaching, or scrambles.",
     "ingredients": "Farm fresh eggs",
     "pairing_keywords": ["sourdough bread", "butter", "cheddar cheese", "pancake mix", "avocado"]},
    {"id": "PROD_024", "name": "Whole Milk Organic (1 Gallon)", "category": "Dairy & Eggs", "price": 4.79,
     "description": "Pasteurized homogenized organic whole vitamin D cow milk from pasture-grazed dairy herds. Rich, creamy, and wholesome.",
     "ingredients": "Organic grade A whole milk, vitamin D3",
     "pairing_keywords": ["rolled oats", "pancake mix", "coffee", "cookies", "cereal"]},
    {"id": "PROD_025", "name": "Creamy Barista Oat Milk", "category": "Dairy & Eggs", "price": 4.29,
     "description": "Specially formulated barista oat milk that steams and micro-foams like dairy milk. Naturally sweet, neutral flavor for lattes and iced coffee.",
     "ingredients": "Oat base (water, oats), low erucic acid rapeseed oil, dipotassium phosphate, sea salt",
     "pairing_keywords": ["espresso beans", "ground coffee", "matcha green tea", "rolled oats", "vanilla syrup"]},
    {"id": "PROD_026", "name": "Plain Whole Milk Greek Yogurt", "category": "Dairy & Eggs", "price": 4.49,
     "description": "Triple-strained thick and velvety Greek yogurt containing live and active probiotics. High protein, rich texture, zero added sugar.",
     "ingredients": "Grade A pasteurized milk, cream, live active cultures",
     "pairing_keywords": ["wildflower honey", "berries", "rolled oats", "granola", "chia seeds"]},
    {"id": "PROD_027", "name": "Sharp Cheddar Cheese Block", "category": "Dairy & Eggs", "price": 4.19,
     "description": "Naturally aged 9-month sharp cheddar cheese block with bold, punchy flavor. Slices clean and melts creamy for grilled sandwiches and burgers.",
     "ingredients": "Cultured pasteurized milk, salt, enzymes, annatto color",
     "pairing_keywords": ["crackers", "sourdough bread", "eggs", "macaroni", "apples"]},
    {"id": "PROD_028", "name": "Heavy Whipping Cream (Pint)", "category": "Dairy & Eggs", "price": 3.29,
     "description": "Ultra-pasteurized heavy whipping cream with 36% milkfat. Whips into stiff, luscious dessert peaks or enriches savory pasta sauces and soups.",
     "ingredients": "Heavy cream, gellan gum",
     "pairing_keywords": ["alfredo sauce", "strawberries", "coffee", "pie", "fettuccine"]},

    # 4. Coffee & Tea Bar
    {"id": "PROD_029", "name": "Italian Roast Whole Bean Espresso", "category": "Coffee & Tea", "price": 12.99,
     "description": "Dark roast 100% Arabica whole bean blend with notes of dark baker's cocoa and toasted caramelized sugars. Produces thick crema.",
     "ingredients": "100% Arabica roasted whole coffee beans",
     "pairing_keywords": ["oat milk", "vanilla syrup", "croissants", "creamer", "biscotti"]},
    {"id": "PROD_030", "name": "Medium Roast Ground Colombian Coffee", "category": "Coffee & Tea", "price": 9.99,
     "description": "Smooth, balanced medium roast single-origin Colombian ground coffee. Bright citrus highlights with a clean nutty finish.",
     "ingredients": "100% Colombian ground coffee",
     "pairing_keywords": ["whole milk", "oat milk", "hazelnut creamer", "sugar", "muffins"]},
    {"id": "PROD_031", "name": "Madagascar Vanilla Coffee Syrup", "category": "Coffee & Tea", "price": 5.99,
     "description": "Artisan barista vanilla flavoring syrup made with pure Madagascar Bourbon vanilla extract and pure cane sugar.",
     "ingredients": "Pure cane sugar, water, natural Madagascar vanilla extract, citric acid",
     "pairing_keywords": ["espresso beans", "oat milk", "cold brew", "latte", "ground coffee"]},
    {"id": "PROD_032", "name": "Sweet Cream Hazelnut Coffee Creamer", "category": "Coffee & Tea", "price": 3.89,
     "description": "Rich liquid coffee creamer infused with warm toasted hazelnut aroma. Blends instantly into hot or iced coffee.",
     "ingredients": "Water, sugar, vegetable oil, sodium caseinate, natural hazelnut flavor",
     "pairing_keywords": ["ground coffee", "espresso beans", "cold brew", "tea", "bagels"]},
    {"id": "PROD_033", "name": "Ceremonial Grade Japanese Matcha", "category": "Coffee & Tea", "price": 16.99,
     "description": "First harvest stone-ground Japanese Uji ceremonial green tea matcha powder. Vibrant emerald green color, delicate umami sweetness.",
     "ingredients": "100% Japanese green tea matcha powder",
     "pairing_keywords": ["oat milk", "vanilla syrup", "honey", "chia seeds", "smoothies"]},
    {"id": "PROD_034", "name": "English Breakfast Black Tea (50 Bags)", "category": "Coffee & Tea", "price": 4.99,
     "description": "Full-bodied robust blend of Assam and Ceylon black tea leaves. High caffeine kick, perfectly complemented by a splash of milk and honey.",
     "ingredients": "Black tea leaves",
     "pairing_keywords": ["whole milk", "raw wildflower honey", "shortbread cookies", "lemon"]},

    # 5. Mexican Fiesta
    {"id": "PROD_035", "name": "White Corn Street Taco Tortillas", "category": "Mexican Fiesta", "price": 2.29,
     "description": "Stone-ground white corn street taco tortillas. Pliable, fragrant, and authentic when warmed on a hot cast iron comal.",
     "ingredients": "Whole grain corn, water, trace of lime",
     "pairing_keywords": ["salsa verde", "guacamole", "sour cream", "black beans", "mexican cheese"]},
    {"id": "PROD_036", "name": "Soft Flour Burrito Tortillas", "category": "Mexican Fiesta", "price": 2.79,
     "description": "Large, soft, and flexible flour tortillas designed to wrap generous burritos and quesadillas without tearing.",
     "ingredients": "Enriched bleached flour, water, vegetable shortening, salt, baking powder",
     "pairing_keywords": ["black beans", "shredded mexican cheese", "sour cream", "salsa", "guacamole"]},
    {"id": "PROD_037", "name": "Tomatillo Salsa Verde Medium", "category": "Mexican Fiesta", "price": 3.49,
     "description": "Bright, zesty salsa made with roasted green tomatillos, fire-roasted serrano peppers, cilantro, lime juice, and garlic.",
     "ingredients": "Tomatillos, serrano peppers, cilantro, onions, lime juice, sea salt, garlic",
     "pairing_keywords": ["tortilla chips", "corn tortillas", "sour cream", "guacamole", "tacos"]},
    {"id": "PROD_038", "name": "Mild Chunky Tomato Salsa", "category": "Mexican Fiesta", "price": 2.99,
     "description": "Thick and chunky homestyle tomato salsa packed with vine-ripened tomatoes, sweet onions, jalapenos, and lime.",
     "ingredients": "Crushed tomatoes, diced onions, jalapenos, cilantro, vinegar, salt, garlic",
     "pairing_keywords": ["tortilla chips", "quesadilla", "tacos", "sour cream", "guacamole"]},
    {"id": "PROD_039", "name": "Fresh Hass Avocado Guacamole", "category": "Mexican Fiesta", "price": 4.99,
     "description": "Chunky restaurant-style guacamole made with ripe Hass avocados, diced red onion, fresh lime juice, cilantro, and sea salt.",
     "ingredients": "Hass avocados, tomatoes, red onions, cilantro, jalapeno peppers, lime juice, salt",
     "pairing_keywords": ["tortilla chips", "corn tortillas", "salsa", "sour cream", "tacos"]},
    {"id": "PROD_040", "name": "Organic Black Beans in Sea Salt", "category": "Mexican Fiesta", "price": 1.49,
     "description": "Creamy organic black beans slow-cooked with sea salt. Rich in plant protein and dietary fiber for tacos, burrito bowls, and salads.",
     "ingredients": "Organic black beans, water, sea salt",
     "pairing_keywords": ["corn tortillas", "jasmine rice", "salsa verde", "guacamole", "mexican cheese"]},
    {"id": "PROD_041", "name": "Authentic Street Taco Seasoning", "category": "Mexican Fiesta", "price": 1.29,
     "description": "Aromatic blend of smoked paprika, toasted cumin, Mexican oregano, garlic powder, onion powder, and chili pepper.",
     "ingredients": "Chili pepper, cumin, salt, smoked paprika, garlic, onion, oregano",
     "pairing_keywords": ["ground beef", "corn tortillas", "black beans", "sour cream", "salsa"]},
    {"id": "PROD_042", "name": "Traditional Cultured Sour Cream", "category": "Mexican Fiesta", "price": 2.19,
     "description": "Grade A pasteurized cultured sour cream with thick, velvety consistency and clean tart flavor to balance spicy tacos and chili.",
     "ingredients": "Cultured cream, enzymes",
     "pairing_keywords": ["salsa", "guacamole", "tortilla chips", "tacos", "baked potato"]},
    {"id": "PROD_043", "name": "Finely Shredded Mexican 4-Cheese Blend", "category": "Mexican Fiesta", "price": 3.99,
     "description": "Gourmet melt blend of Monterey Jack, Mild Cheddar, Asadero, and Queso Quesadilla cheeses. Melts evenly across nachos and burritos.",
     "ingredients": "Monterey Jack, Cheddar, Asadero, Queso Quesadilla, potato starch",
     "pairing_keywords": ["tortilla chips", "corn tortillas", "salsa", "black beans", "sour cream"]},
    {"id": "PROD_044", "name": "Crispy Yellow Corn Tortilla Chips", "category": "Mexican Fiesta", "price": 3.29,
     "description": "Restaurant-style stone-ground yellow corn tortilla chips fried crisp in sunflower oil and dusted with sea salt.",
     "ingredients": "Corn, sunflower oil, sea salt",
     "pairing_keywords": ["salsa verde", "guacamole", "mild chunky salsa", "sour cream", "mexican cheese"]},

    # 6. Asian Staples
    {"id": "PROD_045", "name": "Fragrant Thai Jasmine Rice (5 lb)", "category": "Asian Staples", "price": 7.99,
     "description": "Long-grain aromatic Thai Hom Mali jasmine rice with naturally fragrant pandan aroma and tender, slightly sticky texture.",
     "ingredients": "100% Thai jasmine rice",
     "pairing_keywords": ["soy sauce", "toasted sesame oil", "curry paste", "tofu", "sriracha"]},
    {"id": "PROD_046", "name": "Japanese Ramen Curly Noodles", "category": "Asian Staples", "price": 2.99,
     "description": "Chewy wavy wheat noodles crafted for capturing broth. Boils in 3 minutes for quick ramen bowls or stir-fry yakisoba.",
     "ingredients": "Wheat flour, water, kansui, salt",
     "pairing_keywords": ["soy sauce", "toasted sesame oil", "tofu", "green onions", "sriracha"]},
    {"id": "PROD_047", "name": "Naturally Brewed Tamari Soy Sauce", "category": "Asian Staples", "price": 3.79,
     "description": "Gluten-free traditional Japanese tamari soy sauce naturally fermented for months. Deep, rich umami flavor without artificial additives.",
     "ingredients": "Water, non-GMO soybeans, salt, alcohol",
     "pairing_keywords": ["jasmine rice", "ramen noodles", "toasted sesame oil", "tofu", "stir fry"]},
    {"id": "PROD_048", "name": "Pure Toasted Sesame Oil", "category": "Asian Staples", "price": 4.69,
     "description": "First press dark toasted sesame oil with deep amber hue and intensely fragrant roasted nutty aroma. Essential finishing oil.",
     "ingredients": "100% roasted sesame seed oil",
     "pairing_keywords": ["soy sauce", "ramen noodles", "jasmine rice", "tofu", "sriracha"]},
    {"id": "PROD_049", "name": "Original Huy Fong Sriracha Hot Sauce", "category": "Asian Staples", "price": 4.29,
     "description": "Iconic sun-ripened red jalapeno chili sauce ground into smooth paste with garlic, sugar, and vinegar.",
     "ingredients": "Chili, sugar, salt, garlic, distilled vinegar, potassium sorbate",
     "pairing_keywords": ["ramen noodles", "jasmine rice", "soy sauce", "tofu", "tacos"]},
    {"id": "PROD_050", "name": "Organic Extra Firm Tofu", "category": "Asian Staples", "price": 2.49,
     "description": "High-protein non-GMO organic extra firm soybean curd. Holds its shape when cubed for stir-fries, crispy air frying, or grilling.",
     "ingredients": "Water, organic soybeans, calcium sulfate, nigari",
     "pairing_keywords": ["soy sauce", "toasted sesame oil", "jasmine rice", "ramen noodles", "ginger"]},
    {"id": "PROD_051", "name": "Thai Red Curry Cooking Paste", "category": "Asian Staples", "price": 3.49,
     "description": "Authentic aromatic paste blending dried red chilies, lemongrass, galangal, kaffir lime peel, coriander, and shallots.",
     "ingredients": "Red chili, garlic, lemongrass, shallot, salt, galangal, kaffir lime",
     "pairing_keywords": ["coconut milk", "jasmine rice", "tofu", "vegetables", "soy sauce"]},
    {"id": "PROD_052", "name": "Organic Unsweetened Coconut Milk", "category": "Asian Staples", "price": 2.69,
     "description": "Rich and silky organic full-fat coconut milk pressed from mature coconut flesh. Creamy base for Thai curries and desserts.",
     "ingredients": "Organic coconut milk (coconut, water), organic guar gum",
     "pairing_keywords": ["red curry paste", "jasmine rice", "ramen noodles", "smoothies", "tofu"]},

    # 7. Fresh Produce & Salads
    {"id": "PROD_053", "name": "Crisp Hearts of Romaine Lettuce (3-pack)", "category": "Fresh Produce", "price": 3.49,
     "description": "Sweet, crunchy washed romaine hearts with sturdy leaves. Ideal foundation for classic Caesar salads, wraps, and burger toppings.",
     "ingredients": "Romaine lettuce",
     "pairing_keywords": ["cherry tomatoes", "cucumbers", "caesar dressing", "parmesan cheese", "olive oil"]},
    {"id": "PROD_054", "name": "Sweet Baby Cherry Tomatoes (1 Pint)", "category": "Fresh Produce", "price": 3.29,
     "description": "Vine-ripened sugar sweet cherry tomatoes that burst with juicy sweetness. Perfect for fresh salads, pasta tossing, or roasting.",
     "ingredients": "Cherry tomatoes",
     "pairing_keywords": ["fresh mozzarella", "romaine lettuce", "balsamic vinegar", "extra virgin olive oil", "cucumbers"]},
    {"id": "PROD_055", "name": "Organic Hass Avocados (4-pack)", "category": "Fresh Produce", "price": 4.99,
     "description": "Creamy rich organic Hass avocados with pebbled dark skin and buttery pale-green flesh. High in heart-healthy monounsaturated fats.",
     "ingredients": "Hass avocados",
     "pairing_keywords": ["sourdough bread", "eggs", "tortilla chips", "salsa", "lime"]},
    {"id": "PROD_056", "name": "Crisp English Seedless Cucumbers", "category": "Fresh Produce", "price": 1.79,
     "description": "Tender thin-skinned sweet seedless English cucumber. Mild, hydrating, and crisp for Mediterranean salads and dips.",
     "ingredients": "English cucumber",
     "pairing_keywords": ["cherry tomatoes", "romaine lettuce", "feta cheese", "olive oil", "hummus"]},
    {"id": "PROD_057", "name": "Fresh Organic Baby Spinach (16 oz)", "category": "Fresh Produce", "price": 4.19,
     "description": "Triple-washed tender organic baby spinach leaves. Sweet, mild flavor delicious raw in salads or sautéed with garlic and olive oil.",
     "ingredients": "Organic baby spinach",
     "pairing_keywords": ["cherry tomatoes", "olive oil", "eggs", "parmesan cheese", "strawberries"]},
    {"id": "PROD_058", "name": "Organic Yellow Lemons (2 lb Bag)", "category": "Fresh Produce", "price": 3.99,
     "description": "Juicy bright yellow organic lemons with fragrant citrus zest. Indispensable for dressings, marinades, seafood, and hot tea.",
     "ingredients": "Organic lemons",
     "pairing_keywords": ["green tea", "olive oil", "fresh fish", "salad dressing", "sparkling water"]},
    {"id": "PROD_059", "name": "Fresh Garlic Bulbs (3-pack)", "category": "Fresh Produce", "price": 1.69,
     "description": "Firm plump garlic bulbs with papery white skins and aromatic cloves. The foundational aromatic base for sauces and stir-fries.",
     "ingredients": "Fresh garlic",
     "pairing_keywords": ["olive oil", "pasta", "marinara sauce", "ginger", "soy sauce"]},
    {"id": "PROD_060", "name": "Fresh Cut Basil Herb Pack", "category": "Fresh Produce", "price": 2.29,
     "description": "Fragrant sweet Italian Genovese basil sprigs. Pungent clove-anise aromatics essential for topping hot pizza and pasta dishes.",
     "ingredients": "Fresh sweet basil",
     "pairing_keywords": ["cherry tomatoes", "fresh mozzarella", "marinara sauce", "pasta", "olive oil"]},

    # 8. Snack Attack
    {"id": "PROD_061", "name": "Kettle Cooked Sea Salt Potato Chips", "category": "Snacks & Munchies", "price": 3.69,
     "description": "Thick-cut russet potatoes kettle-cooked in small batches for extra crunch and lightly seasoned with pure sea salt.",
     "ingredients": "Potatoes, avocado oil or sunflower oil, sea salt",
     "pairing_keywords": ["sparkling water", "soda", "sour cream dip", "sandwiches", "beer"]},
    {"id": "PROD_062", "name": "72% Dark Chocolate Cocoa Bar", "category": "Snacks & Munchies", "price": 2.99,
     "description": "Rich artisanal bittersweet Swiss dark chocolate crafted with fair-trade roasted cacao beans. Notes of red fruit and espresso.",
     "ingredients": "Cocoa mass, cocoa butter, organic cane sugar, vanilla extract",
     "pairing_keywords": ["espresso beans", "red wine", "almonds", "rolled oats", "raspberries"]},
    {"id": "PROD_063", "name": "Whole Roasted Salted California Almonds", "category": "Snacks & Munchies", "price": 6.99,
     "description": "Dry-roasted California whole almonds tossed with delicate crystals of sea salt. Satisfying plant protein and crunchy snack.",
     "ingredients": "Almonds, sea salt",
     "pairing_keywords": ["dark chocolate", "dried cranberries", "rolled oats", "granola", "cheese"]},
    {"id": "PROD_064", "name": "Movie Theater Butter Microwave Popcorn", "category": "Snacks & Munchies", "price": 3.79,
     "description": "Fluffy tender whole grain popcorn kernels with rich movie-theatre-style buttery flavor and savory salt seasoning.",
     "ingredients": "Whole grain corn, palm oil, salt, natural butter flavor, annatto",
     "pairing_keywords": ["dark chocolate", "soda", "sparkling water", "gummy candy"]},
    {"id": "PROD_065", "name": "Creamy Roasted Garlic Classic Hummus", "category": "Snacks & Munchies", "price": 3.49,
     "description": "Silky smooth Mediterranean dip made with cooked chickpeas, tahini sesame paste, roasted garlic, cold-pressed oil, and lemon.",
     "ingredients": "Cooked chickpeas, sesame tahini, roasted garlic, sunflower oil, lemon juice, salt",
     "pairing_keywords": ["pretzel crisps", "cucumbers", "cherry tomatoes", "pita bread", "carrots"]},
    {"id": "PROD_066", "name": "Everything Spice Pretzel Crisps", "category": "Snacks & Munchies", "price": 3.49,
     "description": "Thin, flat, crunchy pretzel crackers coated with everything bagel seasoning: sesame seeds, poppy seeds, garlic, onion, and coarse salt.",
     "ingredients": "Enriched wheat flour, sesame seeds, poppy seeds, dehydrated garlic and onion, salt",
     "pairing_keywords": ["roasted garlic hummus", "cream cheese", "cheddar cheese", "dip"]},

    # 9. Beverages
    {"id": "PROD_067", "name": "Sparkling Italian Mineral Water (1 Liter)", "category": "Beverages", "price": 2.19,
     "description": "Naturally effervescent Italian mineral water bottled at the natural spring source. Crisp, refreshing fine mineral carbonation.",
     "ingredients": "Natural mineral water, natural carbon dioxide",
     "pairing_keywords": ["fresh lemon", "potato chips", "wine", "pasta dinner", "espresso"]},
    {"id": "PROD_068", "name": "100% Cold Pressed Valencia Orange Juice", "category": "Beverages", "price": 4.99,
     "description": "Pure freshly squeezed non-from-concentrate Valencia orange juice with high pulp. High vitamin C with natural citrus sweetness.",
     "ingredients": "100% pasteurized orange juice",
     "pairing_keywords": ["pancake mix", "eggs", "croissants", "bacon", "breakfast"]},
    {"id": "PROD_069", "name": "Spicy Ginger Ale Craft Soda (6-pack)", "category": "Beverages", "price": 5.49,
     "description": "Artisan brewed ginger ale made with fresh real ginger root, pure cane sugar, and carbonated mountain water. Refreshing bite.",
     "ingredients": "Carbonated water, cane sugar, ginger root extract, citric acid",
     "pairing_keywords": ["lime", "potato chips", "popcorn", "tacos", "cocktails"]},
    {"id": "PROD_070", "name": "Organic Ginger Lemon Kombucha", "category": "Beverages", "price": 3.69,
     "description": "Raw fermented live probiotic tea infused with fresh pressed ginger and organic lemon juice. Tangy, sparkling gut-health drink.",
     "ingredients": "Brewed organic tea, organic cane sugar, live kombucha culture, ginger juice, lemon juice",
     "pairing_keywords": ["salad greens", "hummus", "almonds", "lunch wrap"]},

    # 10. Household & Cleaning
    {"id": "PROD_071", "name": "Ultra Concentrated Citrus Dish Soap", "category": "Household & Essentials", "price": 3.49,
     "description": "Plant-based grease-cutting dishwashing liquid soap with natural lemon and orange essential oils. Gentle on hands.",
     "ingredients": "Water, plant-derived surfactants, citric acid, natural citrus oils",
     "pairing_keywords": ["paper towels", "sponge", "kitchen cleaner", "trash bags"]},
    {"id": "PROD_072", "name": "Durable Select-A-Size Paper Towels (6 Rolls)", "category": "Household & Essentials", "price": 8.99,
     "description": "2-ply highly absorbent paper towels engineered with trap-and-lock texture for quick kitchen and spill cleanups.",
     "ingredients": "Recycled paper pulp",
     "pairing_keywords": ["citrus dish soap", "all-purpose cleaner", "trash bags"]},
    {"id": "PROD_073", "name": "Eco-Friendly All-Purpose Surface Cleaner", "category": "Household & Essentials", "price": 4.29,
     "description": "Non-toxic biodegradable surface spray cleaner that cuts grime and food residues on countertops, glass, and stove tops.",
     "ingredients": "Purified water, decyl glucoside, lavender essential oil, sodium citrate",
     "pairing_keywords": ["paper towels", "dish soap", "microfiber cloth"]},
    {"id": "PROD_074", "name": "Heavy Duty Kitchen Drawstring Trash Bags (40 ct)", "category": "Household & Essentials", "price": 9.49,
     "description": "13-gallon tear-resistant kitchen garbage bags with reinforced drawstring handles and odor-neutralizing shield.",
     "ingredients": "High-density polyethylene, odor neutralizer",
     "pairing_keywords": ["paper towels", "dish soap", "all-purpose cleaner"]}
]

# Basket Archetypes (Culinary & Behavioral Purchasing Clusters)
BASKET_ARCHETYPES = [
    {
        "name": "Italian Pasta Feast",
        "weight": 0.22,
        "core_items": ["PROD_001", "PROD_004", "PROD_007", "PROD_010"], # Penne, Marinara, Parmigiano, Garlic Baguette
        "optional_items": ["PROD_009", "PROD_008", "PROD_060", "PROD_067", "PROD_011"], # Olive Oil, Mozzarella, Basil, Sparkling Water, Sundried Tomatoes
        "core_prob": 0.85
    },
    {
        "name": "Classic Spaghetti Dinner",
        "weight": 0.16,
        "core_items": ["PROD_002", "PROD_004", "PROD_007"], # Spaghetti, Marinara, Parmigiano
        "optional_items": ["PROD_010", "PROD_009", "PROD_067", "PROD_062"], # Garlic Bread, Olive oil, Sparkling Water, Dark Chocolate
        "core_prob": 0.82
    },
    {
        "name": "Fettuccine Alfredo Night",
        "weight": 0.12,
        "core_items": ["PROD_003", "PROD_005", "PROD_007", "PROD_015"], # Fettuccine, Alfredo, Parmigiano, Butter
        "optional_items": ["PROD_010", "PROD_028", "PROD_067"], # Garlic Bread, Heavy Cream, Sparkling Water
        "core_prob": 0.80
    },
    {
        "name": "Sunday Fluffy Pancake Breakfast",
        "weight": 0.18,
        "core_items": ["PROD_013", "PROD_014", "PROD_015", "PROD_023"], # Pancake Mix, Maple Syrup, Butter, Eggs
        "optional_items": ["PROD_068", "PROD_024", "PROD_030", "PROD_019"], # Orange Juice, Milk, Coffee, Strawberry Spread
        "core_prob": 0.88
    },
    {
        "name": "Morning Espresso & Bakery",
        "weight": 0.14,
        "core_items": ["PROD_029", "PROD_025", "PROD_018"], # Espresso Beans, Oat Milk, Croissants
        "optional_items": ["PROD_031", "PROD_019", "PROD_015"], # Vanilla Syrup, Strawberry Spread, Butter
        "core_prob": 0.80
    },
    {
        "name": "NY Bagel & Cream Cheese Breakfast",
        "weight": 0.12,
        "core_items": ["PROD_020", "PROD_021", "PROD_030"], # Bagels, Cream Cheese, Ground Coffee
        "optional_items": ["PROD_032", "PROD_068", "PROD_023"], # Hazelnut Creamer, Orange Juice, Eggs
        "core_prob": 0.82
    },
    {
        "name": "Mexican Taco Fiesta",
        "weight": 0.20,
        "core_items": ["PROD_035", "PROD_037", "PROD_039", "PROD_043", "PROD_044"], # Corn Tortillas, Salsa Verde, Guacamole, Shredded Cheese, Tortilla Chips
        "optional_items": ["PROD_040", "PROD_042", "PROD_041", "PROD_055"], # Black Beans, Sour Cream, Taco Seasoning, Avocados
        "core_prob": 0.84
    },
    {
        "name": "Burrito Bowl & Nachos",
        "weight": 0.15,
        "core_items": ["PROD_036", "PROD_038", "PROD_040", "PROD_043"], # Burrito Tortillas, Chunky Salsa, Black Beans, Shredded Cheese
        "optional_items": ["PROD_044", "PROD_042", "PROD_039", "PROD_045"], # Chips, Sour Cream, Guacamole, Jasmine Rice
        "core_prob": 0.81
    },
    {
        "name": "Asian Noodle & Stir Fry",
        "weight": 0.15,
        "core_items": ["PROD_046", "PROD_047", "PROD_048", "PROD_050"], # Ramen Noodles, Soy Sauce, Sesame Oil, Tofu
        "optional_items": ["PROD_049", "PROD_059", "PROD_045"], # Sriracha, Fresh Garlic, Jasmine Rice
        "core_prob": 0.83
    },
    {
        "name": "Thai Coconut Curry",
        "weight": 0.10,
        "core_items": ["PROD_045", "PROD_051", "PROD_052", "PROD_050"], # Jasmine Rice, Red Curry Paste, Coconut Milk, Tofu
        "optional_items": ["PROD_047", "PROD_049", "PROD_059"], # Soy Sauce, Sriracha, Garlic
        "core_prob": 0.80
    },
    {
        "name": "Fresh Mediterranean Salad",
        "weight": 0.16,
        "core_items": ["PROD_053", "PROD_054", "PROD_056", "PROD_009", "PROD_012"], # Romaine, Cherry Tomatoes, Cucumber, Olive Oil, Balsamic Vinegar
        "optional_items": ["PROD_008", "PROD_055", "PROD_060", "PROD_007"], # Mozzarella, Avocado, Basil, Parmigiano
        "core_prob": 0.85
    },
    {
        "name": "Healthy Yogurt & Oatmeal Bowl",
        "weight": 0.14,
        "core_items": ["PROD_026", "PROD_016", "PROD_022"], # Greek Yogurt, Rolled Oats, Raw Honey
        "optional_items": ["PROD_063", "PROD_025", "PROD_033"], # Almonds, Oat Milk, Matcha
        "core_prob": 0.82
    },
    {
        "name": "Movie Night Munchies",
        "weight": 0.15,
        "core_items": ["PROD_064", "PROD_061", "PROD_062"], # Popcorn, Potato Chips, Dark Chocolate
        "optional_items": ["PROD_069", "PROD_067", "PROD_063"], # Ginger Ale, Sparkling Water, Almonds
        "core_prob": 0.83
    },
    {
        "name": "Healthy Afternoon Snack & Dip",
        "weight": 0.12,
        "core_items": ["PROD_065", "PROD_066", "PROD_056", "PROD_054"], # Hummus, Pretzel Crisps, Cucumbers, Cherry Tomatoes
        "optional_items": ["PROD_070", "PROD_067"], # Kombucha, Sparkling Water
        "core_prob": 0.84
    },
    {
        "name": "Weekend Household Restock",
        "weight": 0.18,
        "core_items": ["PROD_071", "PROD_072", "PROD_073", "PROD_074"], # Dish Soap, Paper Towels, Cleaner, Trash Bags
        "optional_items": ["PROD_024", "PROD_023", "PROD_017"], # Milk, Eggs, Sourdough Bread
        "core_prob": 0.78
    }
]

ALL_PRODUCT_IDS = [p["id"] for p in PRODUCT_CATALOG]
PROD_LOOKUP = {p["id"]: p for p in PRODUCT_CATALOG}

def generate_products_df():
    """Returns products dataframe."""
    return pd.DataFrame(PRODUCT_CATALOG)

def generate_transactions(num_transactions=50000, start_date="2025-01-01"):
    """
    Generates realistic market basket transactions based on archetypes and background purchasing noise.
    Returns DataFrame with columns: ['transaction_id', 'customer_id', 'timestamp', 'item_count', 'items']
    """
    print(f"Generating {num_transactions:,} market basket transactions...")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    archetype_names = [a["name"] for a in BASKET_ARCHETYPES]
    archetype_weights = [a["weight"] for a in BASKET_ARCHETYPES]
    archetype_weights = np.array(archetype_weights) / sum(archetype_weights)

    records = []
    # Pre-generate random variables for speed
    archetype_choices = np.random.choice(len(BASKET_ARCHETYPES), size=num_transactions, p=archetype_weights)
    customer_ids = [f"CUST_{np.random.randint(1000, 9999):04d}" for _ in range(num_transactions)]
    second_archetype_prob = np.random.random(size=num_transactions)

    for i in range(num_transactions):
        tx_id = f"TX_{i+1:06d}"
        cust_id = customer_ids[i]
        tx_time = start_dt + timedelta(minutes=int(np.random.randint(0, 365 * 24 * 60)))
        
        primary_arch = BASKET_ARCHETYPES[archetype_choices[i]]
        basket_set = set()

        # Add core items with probability
        for item in primary_arch["core_items"]:
            if random.random() < primary_arch["core_prob"]:
                basket_set.add(item)
        
        # Add optional items with probability
        for item in primary_arch["optional_items"]:
            if random.random() < 0.45:
                basket_set.add(item)

        # 30% of shoppers buy from a secondary cluster (e.g. Pasta Dinner + Snack/Household)
        if second_archetype_prob[i] < 0.32:
            second_arch_idx = np.random.choice(len(BASKET_ARCHETYPES))
            second_arch = BASKET_ARCHETYPES[second_arch_idx]
            for item in second_arch["core_items"]:
                if random.random() < 0.60:
                    basket_set.add(item)

        # Background random noise (impulse buys)
        num_noise = np.random.choice([0, 1, 2], p=[0.60, 0.30, 0.10])
        if num_noise > 0:
            noise_items = random.sample(ALL_PRODUCT_IDS, k=num_noise)
            basket_set.update(noise_items)

        # Ensure minimum 2 items per basket
        while len(basket_set) < 2:
            basket_set.add(random.choice(primary_arch["core_items"]))
            basket_set.add(random.choice(primary_arch["optional_items"]))

        basket_items = sorted(list(basket_set))
        records.append({
            "transaction_id": tx_id,
            "customer_id": cust_id,
            "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
            "item_count": len(basket_items),
            "items": ",".join(basket_items)
        })

    df_tx = pd.DataFrame(records)
    print(f"Generated {len(df_tx):,} transactions with average basket size of {df_tx['item_count'].mean():.2f} items.")
    return df_tx

# Authentic review templates with sentiment and pairing clues
POSITIVE_REVIEW_TEMPLATES = [
    "Absolutely love this {name}! It pairs so wonderfully with {pair1} and {pair2}. The texture and taste are unmatched.",
    "A staple in our weekly grocery cart. Whenever we make {name}, we always serve it alongside {pair1}. Everyone raved about dinner.",
    "Five stars! The quality of this {name} is restaurant grade. Essential companion for {pair1}, especially when topped with {pair2}.",
    "Incredible flavor profile. If you buy this {name}, do not forget to pick up {pair1} as well—they complement each other flawlessly.",
    "Best {name} on the market. We combined it with {pair1} for our weekend meal and it was an absolute hit with the entire family.",
    "Outstanding freshness. The rich flavor pairs perfectly with {pair1} and a sprinkle of {pair2}. Highly recommend!",
    "Great product and superb value! We always stock this {name} along with {pair1}. Can't imagine having one without the other.",
    "Authentic taste and premium quality. Made a quick dinner using this {name} tossed with {pair1} and {pair2}—pure perfection."
]

NEUTRAL_REVIEW_TEMPLATES = [
    "Decent quality {name}. It works well enough when cooked with {pair1}, but flavor is relatively mild compared to other brands.",
    "Good standard {name}. It does the job nicely alongside {pair1}, nothing extraordinary but reliable.",
    "Average price and decent taste. Paired it with {pair1} as suggested; tasted okay for a weeknight quick meal."
]

NEGATIVE_REVIEW_TEMPLATES = [
    "Disappointed with this {name}. Texture was off and didn't blend well with our {pair1}. Expected better quality.",
    "Not worth the price. The flavor of this {name} felt stale and completely ruined our dish with {pair1}.",
    "Mediocre product. Arrived close to expiration date and didn't taste fresh when served with {pair1}."
]

def generate_reviews(num_reviews=15000):
    """Generates authentic customer reviews incorporating sentiment and pairing co-mentions."""
    print(f"Generating {num_reviews:,} customer product reviews...")
    review_records = []
    
    for i in range(num_reviews):
        review_id = f"REV_{i+1:06d}"
        prod = random.choice(PRODUCT_CATALOG)
        prod_id = prod["id"]
        prod_name = prod["name"]
        keywords = prod["pairing_keywords"]
        
        pair1 = keywords[0] if len(keywords) > 0 else "side dishes"
        pair2 = keywords[1] if len(keywords) > 1 else "favorite toppings"
        
        # Sentiment distribution: 75% positive, 15% neutral, 10% negative
        sentiment_bucket = np.random.choice(["positive", "neutral", "negative"], p=[0.78, 0.14, 0.08])
        
        if sentiment_bucket == "positive":
            rating = np.random.choice([4, 5], p=[0.30, 0.70])
            template = random.choice(POSITIVE_REVIEW_TEMPLATES)
            review_text = template.format(name=prod_name, pair1=pair1, pair2=pair2)
        elif sentiment_bucket == "neutral":
            rating = 3
            template = random.choice(NEUTRAL_REVIEW_TEMPLATES)
            review_text = template.format(name=prod_name, pair1=pair1)
        else:
            rating = np.random.choice([1, 2], p=[0.60, 0.40])
            template = random.choice(NEGATIVE_REVIEW_TEMPLATES)
            review_text = template.format(name=prod_name, pair1=pair1)

        verified = random.random() < 0.92
        user_id = f"USER_{np.random.randint(10000, 99999)}"
        created_at = (datetime(2025, 1, 1) + timedelta(days=int(np.random.randint(0, 360)))).strftime("%Y-%m-%d")

        review_records.append({
            "review_id": review_id,
            "product_id": prod_id,
            "user_id": user_id,
            "rating": rating,
            "review_text": review_text,
            "verified_purchase": verified,
            "created_at": created_at
        })

    df_rev = pd.DataFrame(review_records)
    print(f"Generated {len(df_rev):,} reviews with average rating of {df_rev['rating'].mean():.2f}/5.0.")
    return df_rev

def build_and_save_all_datasets(num_transactions=50000, num_reviews=15000, train_ratio=0.80):
    """Orchestrates generation and writes out train/test splits and CSVs."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Products
    df_products = generate_products_df()
    prod_path = os.path.join(DATA_DIR, "products.csv")
    df_products.to_csv(prod_path, index=False)
    print(f"Saved products: {prod_path} ({len(df_products)} rows)")

    # 2. Transactions
    df_tx = generate_transactions(num_transactions=num_transactions)
    tx_path = os.path.join(DATA_DIR, "transactions.csv")
    df_tx.to_csv(tx_path, index=False)
    print(f"Saved all transactions: {tx_path} ({len(df_tx)} rows)")

    # 3. Train/Test Split (80/20)
    shuffled_indices = np.random.permutation(len(df_tx))
    split_point = int(len(df_tx) * train_ratio)
    train_indices = shuffled_indices[:split_point]
    test_indices = shuffled_indices[split_point:]

    df_train = df_tx.iloc[train_indices].reset_index(drop=True)
    df_test = df_tx.iloc[test_indices].reset_index(drop=True)

    train_path = os.path.join(DATA_DIR, "train_transactions.csv")
    test_path = os.path.join(DATA_DIR, "test_transactions.csv")
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    print(f"Saved train transactions: {train_path} ({len(df_train):,} rows)")
    print(f"Saved test transactions: {test_path} ({len(df_test):,} rows)")

    # 4. Reviews
    df_reviews = generate_reviews(num_reviews=num_reviews)
    rev_path = os.path.join(DATA_DIR, "reviews.csv")
    df_reviews.to_csv(rev_path, index=False)
    print(f"Saved customer reviews: {rev_path} ({len(df_reviews):,} rows)")

    return df_products, df_train, df_test, df_reviews

if __name__ == "__main__":
    build_and_save_all_datasets(num_transactions=50000, num_reviews=15000)
