import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
DATA_PATH = os.path.join("data", "sales_data_sample.csv")

DATE_COLUMN = "ORDERDATE"          # change to your date column
PRODUCT_COLUMN = "PRODUCTLINE"     # change to your product/product-line column
QTY_COLUMN = "QUANTITYORDERED"     # change if needed
PRICE_COLUMN = "PRICEEACH"         # change if needed
SALES_COLUMN = "SALES"             # total sales column (or will be computed)


# =========================
# LOAD DATA
# =========================
def load_data(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        # Add encoding here to avoid UnicodeDecodeError
        df = pd.read_csv(path, encoding='latin1')
    elif path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        raise ValueError("Only CSV or Excel files supported.")
    return df


# =========================
# CLEAN & PREPROCESS
# =========================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Drop fully empty rows
    df.dropna(how="all", inplace=True)

    # Convert numeric columns
    if QTY_COLUMN in df.columns:
        df[QTY_COLUMN] = pd.to_numeric(df[QTY_COLUMN], errors="coerce").fillna(0)
    if PRICE_COLUMN in df.columns:
        df[PRICE_COLUMN] = pd.to_numeric(df[PRICE_COLUMN], errors="coerce").fillna(0)

    # Convert date
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
        df = df[df[DATE_COLUMN].notna()]

    # Create SALES if missing
    if SALES_COLUMN not in df.columns and all(
        col in df.columns for col in [QTY_COLUMN, PRICE_COLUMN]
    ):
        df[SALES_COLUMN] = df[QTY_COLUMN] * df[PRICE_COLUMN]

    # Clean SALES
    if SALES_COLUMN in df.columns:
        df[SALES_COLUMN] = pd.to_numeric(df[SALES_COLUMN], errors="coerce").fillna(0)
        df = df[df[SALES_COLUMN] > 0]

    return df


# =========================
# TIME FEATURES
# =========================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if DATE_COLUMN not in df.columns:
        raise KeyError(f"{DATE_COLUMN} column is required for time analysis")

    df["Year"] = df[DATE_COLUMN].dt.year
    df["Month"] = df[DATE_COLUMN].dt.month
    df["YearMonth"] = df[DATE_COLUMN].dt.to_period("M").astype(str)
    return df


# =========================
# ANALYSIS
# =========================
def get_top_products(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if PRODUCT_COLUMN not in df.columns:
        raise KeyError(f"{PRODUCT_COLUMN} column is required for product analysis")

    top_products = (
        df.groupby(PRODUCT_COLUMN)[SALES_COLUMN]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    return top_products


def get_monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    if "YearMonth" not in df.columns:
        raise KeyError("YearMonth column is missing. Run add_time_features first.")

    monthly_sales = (
        df.groupby("YearMonth")[SALES_COLUMN]
        .sum()
        .reset_index()
        .sort_values("YearMonth")
    )
    return monthly_sales


def get_basic_stats(df: pd.DataFrame) -> dict:
    stats = {
        "total_rows": int(df.shape[0]),
        "total_revenue": float(df[SALES_COLUMN].sum()),
        "avg_order_value": float(df[SALES_COLUMN].mean()),
    }
    if PRODUCT_COLUMN in df.columns:
        stats["unique_products"] = int(df[PRODUCT_COLUMN].nunique())
    else:
        stats["unique_products"] = None
    return stats


# =========================
# VISUALIZATION
# =========================
def plot_monthly_trends(monthly_sales: pd.DataFrame) -> None:
    if monthly_sales.empty:
        print("No monthly sales data to plot.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(monthly_sales["YearMonth"], monthly_sales[SALES_COLUMN], marker="o")
    plt.title("Monthly Sales Trend")
    plt.xlabel("Year-Month")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_top_products(top_products: pd.DataFrame) -> None:
    if top_products.empty:
        print("No top products data to plot.")
        return

    plt.figure(figsize=(10, 5))
    plt.bar(top_products[PRODUCT_COLUMN].astype(str), top_products[SALES_COLUMN])
    plt.title("Top Products by Total Sales")
    plt.xlabel("Product")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# =========================
# TEXT REPORT
# =========================
def print_report(df: pd.DataFrame,
                 stats: dict,
                 top_products: pd.DataFrame,
                 monthly_sales: pd.DataFrame) -> None:
    print("=" * 60)
    print("BASIC SALES ANALYSIS REPORT")
    print("=" * 60)

    print(f"Total rows (order lines): {stats['total_rows']}")
    print(f"Total revenue           : {stats['total_revenue']:,.2f}")
    print(f"Average order value     : {stats['avg_order_value']:,.2f}")
    if stats["unique_products"] is not None:
        print(f"Unique products         : {stats['unique_products']}")
    print()

    if not top_products.empty:
        print("Top 5 products by total sales:")
        for _, row in top_products.head(5).iterrows():
            print(f"- {row[PRODUCT_COLUMN]}: {row[SALES_COLUMN]:,.2f}")
        print()

    if not monthly_sales.empty:
        best = monthly_sales.loc[monthly_sales[SALES_COLUMN].idxmax()]
        worst = monthly_sales.loc[monthly_sales[SALES_COLUMN].idxmin()]
        print("Monthly sales summary:")
        print(f"- Best month : {best['YearMonth']}  (Sales: {best[SALES_COLUMN]:,.2f})")
        print(f"- Worst month: {worst['YearMonth']} (Sales: {worst[SALES_COLUMN]:,.2f})")
        print()

    print("End of report.")
    print("=" * 60)


# =========================
# MAIN
# =========================
def main() -> None:
    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Clean & preprocess
    df = clean_data(df)

    # 3. Add time features
    df = add_time_features(df)

    # 4. Analysis
    stats = get_basic_stats(df)
    top_products = get_top_products(df, top_n=10)
    monthly_sales = get_monthly_trends(df)

    # 5. Report
    print_report(df, stats, top_products, monthly_sales)

    # 6. Plots
    plot_monthly_trends(monthly_sales)
    plot_top_products(top_products)


if __name__ == "__main__":
    main()
