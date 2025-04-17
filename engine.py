import pandas as pd

class RecommendationEngine:
    def __init__(self):
        self.main_df = pd.read_csv(r"D:\projects\old\MobiGenie\data\main_product_data.csv")
        self.laptop_df = pd.read_csv(r"D:\projects\old\MobiGenie\data\laptop_product_accessories.csv")
        self.mobile_df = pd.read_csv(r"D:\projects\old\MobiGenie\data\mobile_product_accessories.csv")

    def get_top_products(self, brand, category, top_n=5):
        df = self.main_df
        filtered = df[
            (df['brand'].str.lower() == brand.lower()) & 
            (df['category'].str.lower() == category.lower())
        ]
        top = filtered.sort_values(by='rating', ascending=False).head(top_n)
        return top.to_dict(orient='records')

    def get_accessories(self, brand, category):
        df = self.laptop_df if category.lower() == "laptop" else self.mobile_df
        acc = df[df['brand'].str.lower() == brand.lower()]
        return acc.to_dict(orient='records')
