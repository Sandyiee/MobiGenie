from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import sys
import os
import random

# For Cosine Similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

# Load Recommender + Chatbot
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'recommender'))
from recommendation.engine import RecommendationEngine
from chatbot.mistral_integration import MistralChatbot

recommender = RecommendationEngine()
chatbot = MistralChatbot()

# Helper: Safe Conversion
def safe_value(val, cast_type):
    try:
        return cast_type(val)
    except:
        return cast_type(0)

# Pydantic Models
class NameRequest(BaseModel):
    name: str

class ChatRequest(BaseModel):
    query: str

class ModelRequest(BaseModel):
    model: str

@app.post("/greet_user")
def greet_user(req: NameRequest):
    name = req.name
    try:
        df = recommender.main_df
        laptops = df[df['category'].str.lower() == 'laptops']
        mobiles = df[df['category'].str.lower() == 'smartphones']

        unique_laptops = laptops['model'].dropna().unique().tolist()
        unique_mobiles = mobiles['model'].dropna().unique().tolist()

        greeting = f"""
Hi {name}! 👋 I'm MobiGenie, your smart shopping buddy.

🎉 Welcome to MobiGenie Store! Let’s explore some gadgets.

💻 Laptops:
{', '.join(unique_laptops)}

📱 Smartphones:
{', '.join(unique_mobiles)}

Let me know what you're interested in! 😊
"""
        return {"response": greeting}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask_mistral")
def ask_mistral(req: ChatRequest):
    try:
        response = chatbot.ask_question(req.query)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

@app.post("/get_model_details")
def get_model_details(req: ModelRequest):
    model_input = req.model.strip().lower()
    print(f"🔍 Model requested: {model_input}")

    try:
        df = recommender.main_df
        titles = df['model'].fillna('').astype(str)
        tfidf = TfidfVectorizer().fit(titles)
        title_vectors = tfidf.transform(titles)

        query_vec = tfidf.transform([model_input])
        cosine_sim = cosine_similarity(query_vec, title_vectors).flatten()
        top_match_idx = cosine_sim.argmax()
        top_score = cosine_sim[top_match_idx]

        if top_score < 0.3:
            return {"error": f"No similar product found for model: {req.model}"}

        product = df.iloc[top_match_idx]
        print(f"✅ Best match: {product['model']} (score: {top_score:.2f})")

        category = product['category']
        brand = product['brand']
        price = safe_value(product['price'], float)
        rating = safe_value(product['rating'], float)
        review_count = safe_value(product['review_count'], int)
        title = str(product['product_title'])
        image_url = str(product['image_url'])
        stars = "⭐" * int(round(rating))

        main_product_info = {
            "title": title,
            "price": price,
            "rating": rating,
            "review_count": review_count,
            "stars": stars,
            "image_url": image_url
        }

        df_filtered = df[
            (df['category'] == category) &
            (df['price'].astype(float) >= price * 0.8) &
            (df['price'].astype(float) <= price * 1.2)
        ]

        same_brand = df_filtered[df_filtered['brand'].str.lower() == brand.lower()]
        other_brand = df_filtered[df_filtered['brand'].str.lower() != brand.lower()]

        recommendations = []
        for _, row in same_brand.head(2).iterrows():
            recommendations.append(build_product_card("Same Brand Recommendation", row))

        for _, row in other_brand.head(2).iterrows():
            recommendations.append(build_product_card("Other Brand Recommendation", row))

        accessories_info = get_accessories_for_model(model_input, category, brand)
        print(f"✅ Accessories found: {len(accessories_info)}")

        return jsonable_encoder({
            "response": main_product_info,
            "similar_products": recommendations,
            "accessories": accessories_info
        })

    except Exception as e:
        print("❌ Error in get_model_details:", e)
        return {"error": str(e)}

def build_product_card(brand_type, row):
    price = safe_value(row['price'], float)
    rating = safe_value(row['rating'], float)
    review_count = safe_value(row['review_count'], int)
    stars = "⭐" * int(round(rating))
    return {
        "brand_type": brand_type,
        "title": str(row['product_title']),
        "price": price,
        "rating": rating,
        "review_count": review_count,
        "stars": stars,
        "image_url": str(row['image_url'])
    }

def get_accessories_for_model(model_name, category, brand):
    accessories = []
    model_name_lower = model_name.strip().lower()

    if category.lower() == 'smartphones':
        df = recommender.mobile_df

        screen_protector = df[
            (df['category'].str.lower() == 'screen protector') &
            (df['model'].str.lower().str.strip() == model_name_lower)
        ]
        if not screen_protector.empty:
            accessories.append(build_accessory("Screen Protector", screen_protector.iloc[0]))

        phone_case = df[
            (df['category'].str.lower() == 'phone cases') &
            (df['model'].str.lower().str.strip() == model_name_lower)
        ]
        if not phone_case.empty:
            accessories.append(build_accessory("Phone Case", phone_case.iloc[0]))

        if brand.lower() == 'apple':
            chargers = df[df['category'].str.lower() == 'charger']
            if not chargers.empty:
                row = random.choice(chargers.to_dict(orient='records'))
                accessories.append(build_accessory("Charger", row, dict_row=True))

        bt_headphones = df[df['category'].str.lower() == 'bluetooth headphone']
        if not bt_headphones.empty:
            row = random.choice(bt_headphones.to_dict(orient='records'))
            accessories.append(build_accessory("Bluetooth Headphone", row, dict_row=True))

    elif category.lower() == 'laptops':
        df = recommender.laptop_df
        main_df = recommender.main_df

        inch_series = main_df[main_df['model'].str.lower().str.strip() == model_name_lower]['inch']
        if not inch_series.empty:
            try:
                product_inch = float(inch_series.iloc[0])
                laptop_bags = df[
                    (df['category'].str.lower() == 'laptop bags') &
                    (df['inch'].notnull()) &
                    (df['inch'].astype(float) == product_inch)
                ]
                if not laptop_bags.empty:
                    accessories.append(build_accessory(f"Laptop Bag ({product_inch}\" size)", laptop_bags.iloc[0]))
            except ValueError:
                pass

        if brand.lower() == 'apple':
            magic_mouse = df[
                (df['category'].str.lower() == 'mouse') &
                (df['product_title'].str.lower().str.contains("magic mouse"))
            ]
            if not magic_mouse.empty:
                accessories.append(build_accessory("Magic Mouse", magic_mouse.iloc[0]))

        elif brand.lower() == 'samsung':
            other_mouse = df[
                (df['category'].str.lower() == 'mouse') &
                (~df['product_title'].str.lower().str.contains("magic mouse"))
            ]
            if not other_mouse.empty:
                accessories.append(build_accessory("Mouse", other_mouse.iloc[0]))
        else:
            any_mouse = df[df['category'].str.lower() == 'mouse']
            if not any_mouse.empty:
                accessories.append(build_accessory("Mouse", any_mouse.iloc[0]))

    return accessories

def build_accessory(accessory_type, row, dict_row=False):
    if dict_row:
        rating = safe_value(row['rating'], float)
        stars = "⭐" * int(round(rating))
        return {
            "accessory_type": accessory_type,
            "title": str(row['product_title']),
            "price": safe_value(row['price'], float),
            "rating": rating,
            "review_count": safe_value(row['review_count'], int),
            "stars": stars,
            "image_url": str(row['image_url'])
        }
    else:
        price = safe_value(row['price'], float)
        rating = safe_value(row['rating'], float)
        review_count = safe_value(row['review_count'], int)
        stars = "⭐" * int(round(rating))
        return {
            "accessory_type": accessory_type,
            "title": str(row['product_title']),
            "price": price,
            "rating": rating,
            "review_count": review_count,
            "stars": stars,
            "image_url": str(row['image_url'])
        }
