import csv
import random
from datetime import datetime, timedelta

random.seed(42)

# Configuration
NUM_RECORDS = 10000
START_DATE = datetime(2024, 1, 1)

# Categories and products
categories = {
    "Electronics": ["Laptop", "Smartphone", "Tablet", "Headphones", "Smartwatch", "Camera"],
    "Clothing": ["T-Shirt", "Jeans", "Jacket", "Sneakers", "Hat", "Socks"],
    "Home": ["Lamp", "Chair", "Table", "Blender", "Toaster", "Pillow"],
    "Books": ["Novel", "Textbook", "Comic", "Biography", "Cookbook", "Guide"],
    "Sports": ["Football", "Basketball", "Yoga Mat", "Dumbbell", "Tennis Racket", "Bicycle"]
}

regions = ["North", "South", "East", "West", "Central"]
payment_methods = ["Credit Card", "Debit Card", "PayPal", "Mobile Money", "Bank Transfer"]
channels = ["Web", "Mobile App", "In-Store", "Phone"]

# Price ranges per category
price_ranges = {
    "Electronics": (50, 1500),
    "Clothing": (10, 150),
    "Home": (15, 400),
    "Books": (5, 60),
    "Sports": (20, 800)
}

def generate_record(order_id):
    category = random.choice(list(categories.keys()))
    product = random.choice(categories[category])
    min_p, max_p = price_ranges[category]
    
    unit_price = round(random.uniform(min_p, max_p), 2)
    quantity = random.randint(1, 5)
    discount = random.choice([0, 0, 0, 0.05, 0.1, 0.15, 0.2])
    
    subtotal = unit_price * quantity
    discount_amount = round(subtotal * discount, 2)
    total = round(subtotal - discount_amount, 2)
    
    # Random date within 2024
    days_offset = random.randint(0, 364)
    order_date = START_DATE + timedelta(days=days_offset)
    
    # Customer satisfaction (1-5) — higher for discounted items
    base_satisfaction = random.choices([1, 2, 3, 4, 5], weights=[5, 10, 20, 35, 30])[0]
    if discount > 0.1:
        base_satisfaction = min(5, base_satisfaction + 1)
    
    return {
        "order_id": f"ORD{order_id:06d}",
        "order_date": order_date.strftime("%Y-%m-%d"),
        "customer_id": f"CUST{random.randint(1, 2000):05d}",
        "category": category,
        "product": product,
        "unit_price": unit_price,
        "quantity": quantity,
        "discount": discount,
        "total_amount": total,
        "region": random.choice(regions),
        "payment_method": random.choice(payment_methods),
        "channel": random.choice(channels),
        "satisfaction": base_satisfaction,
        "delivery_days": random.randint(1, 14)
    }

# Generate records
records = [generate_record(i) for i in range(1, NUM_RECORDS + 1)]

# Write to CSV
filename = "bigdata_analytics/ecommerce_sales.csv"
with open(filename, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

print(f"Dataset created: {filename}")
print(f"Total records: {len(records)}")
print(f"\nSample records:")
for r in records[:3]:
    print(r)