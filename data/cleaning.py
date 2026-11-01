import pandas as pd
import numpy as np

DATA = "data/"

orders      = pd.read_csv(DATA + "olist_orders_dataset.csv")
customers   = pd.read_csv(DATA + "olist_customers_dataset.csv")
items       = pd.read_csv(DATA + "olist_order_items_dataset.csv")
payments    = pd.read_csv(DATA + "olist_order_payments_dataset.csv")
reviews     = pd.read_csv(DATA + "olist_order_reviews_dataset.csv")
products    = pd.read_csv(DATA + "olist_products_dataset.csv")
sellers     = pd.read_csv(DATA + "olist_sellers_dataset.csv")
translation = pd.read_csv(DATA + "product_category_name_translation.csv")

for name, df in {"orders": orders, "customers": customers, "items": items,
                 "payments": payments, "reviews": reviews,
                 "products": products, "sellers": sellers}.items():
    print(f"{name:10} rows={len(df):>7}  duplicates={df.duplicated().sum()}")

for name, df in {"orders": orders, "products": products, "reviews": reviews}.items():
    print(name)
    print(df.isnull().sum()[df.isnull().sum() > 0], "\n")

date_cols = ["order_purchase_timestamp", "order_approved_at",
             "order_delivered_carrier_date", "order_delivered_customer_date",
             "order_estimated_delivery_date"]
for c in date_cols:
    orders[c] = pd.to_datetime(orders[c])

print(orders["order_status"].value_counts())

# Keep delivered orders for revenue and delivery analysis
delivered = orders[orders["order_status"] == "delivered"].dropna(
    subset=["order_delivered_customer_date"]).copy()

# New columns
delivered["order_month"] = delivered["order_purchase_timestamp"].dt.to_period("M").astype(str)
delivered["delivery_days"] = (delivered["order_delivered_customer_date"]
                              - delivered["order_purchase_timestamp"]).dt.days
delivered["is_late"] = (delivered["order_delivered_customer_date"]
                        > delivered["order_estimated_delivery_date"]).astype(int)
print(delivered.shape)

products = products.merge(translation, on="product_category_name", how="left")
products["category"] = products["product_category_name_english"].fillna("unknown")
products = products[["product_id", "category"]]
products["category"].value_counts().head(10)

reviews["review_creation_date"] = pd.to_datetime(reviews["review_creation_date"])
reviews = (reviews.sort_values("review_creation_date")
                  .drop_duplicates("order_id", keep="last")[["order_id", "review_score"]])

payments_agg = (payments.groupby("order_id")
                .agg(payment_value=("payment_value", "sum"),
                     payment_type=("payment_type", "first"))
                .reset_index())

print(reviews["order_id"].is_unique, payments_agg["order_id"].is_unique)

# Order-level table (one row per order)
order_df = (delivered
            .merge(customers, on="customer_id", how="left")
            .merge(payments_agg, on="order_id", how="left")
            .merge(reviews, on="order_id", how="left"))

# Item-level table (one row per item sold, for category and seller analysis)
item_df = (items
           .merge(products, on="product_id", how="left")
           .merge(delivered[["order_id", "order_month", "customer_id"]], on="order_id")
           .merge(customers[["customer_id", "customer_unique_id", "customer_state"]],
                  on="customer_id", how="left"))

print(order_df.shape, item_df.shape)
print("Total revenue:", round(order_df["payment_value"].sum(), 2))

order_df.to_csv(DATA + "clean_orders.csv", index=False)
item_df.to_csv(DATA + "clean_items.csv", index=False)

print(order_df.groupby("is_late")["review_score"].mean())
print(order_df.groupby("order_month")["payment_value"].sum().tail(6))