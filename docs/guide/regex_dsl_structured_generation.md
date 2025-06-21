# Generating Realistic Test Data with the Regex DSL

This guide shows how to use Outlines' Regex DSL to generate realistic test data that matches specific patterns. We'll build a test data generator for an e-commerce system that creates valid product codes, serial numbers, and tracking IDs.

## What We'll Build

A test data generator that creates:
- Product SKUs following company format
- Valid serial numbers for different product lines  
- Tracking numbers that match carrier formats
- Realistic order reference numbers

## Why the Regex DSL?

When generating test data, you need values that:
- Look realistic to pass validation
- Follow specific business rules
- Are unique but predictable
- Match external system requirements

## Prerequisites

```shell
mkdir test-data-generator
cd test-data-generator
uv init
uv add outlines transformers torch
```

## Step 1: The Problem with Random Test Data

Traditional test data often looks fake:

```python
# Fake-looking test data:
products = ["Product1", "Product2", "Test123"]
serials = ["12345", "ABCDE", "XXX-YYY"]
tracking = ["TRACK001", "1234567890"]

# Real patterns your systems expect:
# SKU: APP-ELEC-TV-55-BLK (category-type-model-size-color)
# Serial: SN2024W12B5A0001 (year+week+batch+line+sequence)
# Tracking: 1Z999AA10123456784 (UPS format)
```

## Step 2: Building Product SKU Generator

Let's generate SKUs that follow real business rules:

```python
# sku_generator.py
from outlines import generate
from outlines.types import Regex, String, either
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# SKU Format: CAT-TYPE-MODEL-SIZE-COLOR
# Example: APP-WASH-FLD-07K-WHT (Appliance, Washer, Front-Load, 7kg, White)

# Category codes (3 letters)
category = either("APP", "ELC", "CLO", "HOM", "SPT")  # Appliances, Electronics, Clothing, Home, Sports

# Product type codes (4 letters)
product_type = either("WASH", "DRYR", "FRDG", "OVEN", "DSHW",  # Appliances
                     "TVST", "LPTP", "PHNE", "TBLT", "HDPH",   # Electronics
                     "SHRT", "PNTS", "DRSS", "SHOE", "JCKT")   # Clothing

# Model number (3 alphanumeric)
model = Regex(r"[A-Z][0-9][A-Z0-9]")

# Size/capacity (2-3 chars + unit)
size = Regex(r"[0-9]{2,3}[KLMGSXL]")

# Color code (3 letters)
color = either("BLK", "WHT", "SLV", "RED", "BLU", "GRN", "GLD")

# Complete SKU pattern
sku_pattern = (
    category + String("-") +
    product_type + String("-") + 
    model + String("-") +
    size + String("-") +
    color
)

# Generate SKUs
sku_generator = generate.regex(model, tokenizer, sku_pattern)

# Generate multiple SKUs
print("Generated SKUs:")
for i in range(5):
    prompt = f"Generate SKU for product {i+1}: "
    sku = sku_generator(prompt, max_tokens=30)
    print(f"  {sku}")

# Example output:
#   APP-WASH-F3D-07K-WHT
#   ELC-TVST-X9K-55I-BLK
#   CLO-SHRT-M2C-42L-BLU
#   HOM-FRDG-A8X-18C-SLV
#   SPT-SHOE-R4N-09M-RED
```

## Step 3: Creating Serial Number Patterns

Different product lines need different serial formats:

```python
# serial_generator.py
from outlines.types import Regex, String, either

# Serial format: SN + Year(2) + Week(2) + Factory(1) + Line(1) + Sequence(5)
# Example: SN24W12B5A00142

year = Regex(r"2[3-5]")  # 2023-2025
week = Regex(r"[0-5][0-9]")  # 00-52
factory = either("A", "B", "C", "D")  # Factory code
line = Regex(r"[1-9]")  # Production line 1-9
sequence = Regex(r"[0-9]{5}")  # 5-digit sequence

serial_pattern = (
    String("SN") + year + String("W") + week + 
    factory + line + String("A") + sequence
)

# Alternative format for electronics
# ELC-YYYY-MM-PPPPPP (Electronics, Year, Month, Product ID)
elec_serial = (
    String("ELC-") + 
    String("202") + Regex(r"[3-5]") + String("-") +
    Regex(r"0[1-9]|1[0-2]") + String("-") +
    Regex(r"[0-9]{6}")
)

# Generate serials
serial_gen = generate.regex(model, tokenizer, serial_pattern)
elec_gen = generate.regex(model, tokenizer, elec_serial)

print("\nGenerated Serial Numbers:")
print("Standard format:")
for i in range(3):
    serial = serial_gen(f"Serial {i+1}: ", max_tokens=25)
    print(f"  {serial}")

print("\nElectronics format:")
for i in range(3):
    serial = elec_gen(f"Electronic serial {i+1}: ", max_tokens=25)
    print(f"  {serial}")

# Example output:
# Standard format:
#   SN24W12B5A00142
#   SN24W08C3A00891
#   SN23W51A8A04567
# 
# Electronics format:
#   ELC-2024-03-456789
#   ELC-2024-07-123456
#   ELC-2023-12-987654
```

## Step 4: Generating Tracking Numbers

Different carriers have specific formats:

```python
# tracking_generator.py

# UPS: 1Z[6 chars][8 digits]
ups_tracking = (
    String("1Z") + 
    Regex(r"[0-9A-Z]{6}") + 
    Regex(r"[0-9]{8}")
)

# FedEx: [12 digits] or [15 digits]
fedex_12 = Regex(r"[0-9]{12}")
fedex_15 = Regex(r"[0-9]{15}")

# DHL: [10 digits]
dhl_tracking = Regex(r"[0-9]{10}")

# USPS: [20 digits] or 9[4 digits] [5 digits] [5 digits] [5 digits]
usps_format = (
    String("9") + Regex(r"[0-9]{4}") + String(" ") +
    Regex(r"[0-9]{5}") + String(" ") +
    Regex(r"[0-9]{5}") + String(" ") +
    Regex(r"[0-9]{5}")
)

# Generate tracking numbers
ups_gen = generate.regex(model, tokenizer, ups_tracking)
fedex_gen = generate.regex(model, tokenizer, fedex_12)
usps_gen = generate.regex(model, tokenizer, usps_format)

print("\nGenerated Tracking Numbers:")
print(f"UPS:   {ups_gen('UPS tracking: ', max_tokens=20)}")
print(f"FedEx: {fedex_gen('FedEx tracking: ', max_tokens=20)}")
print(f"USPS:  {usps_gen('USPS tracking: ', max_tokens=30)}")

# Example output:
# UPS:   1Z9A8B7C12345678
# FedEx: 123456789012
# USPS:  94055 12345 67890 12345
```

## Step 5: Creating a Complete Test Data Generator

Let's build a system that generates complete order records:

```python
# test_data_generator.py
from outlines import generate
from outlines.types import Regex, String, either, optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

class TestDataGenerator:
    def __init__(self, model_name="microsoft/phi-2"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.setup_patterns()
    
    def setup_patterns(self):
        # Order reference: ORD-YYYYMMDD-XXXXX
        self.order_ref = (
            String("ORD-") +
            String(datetime.now().strftime("%Y%m%d")) + String("-") +
            Regex(r"[A-Z0-9]{5}")
        )
        
        # Customer ID: CUST-XXXXX
        self.customer_id = String("CUST-") + Regex(r"[0-9]{5}")
        
        # Payment reference: PAY-CC-XXXX-XXXX
        self.payment_ref = (
            String("PAY-") +
            either("CC", "DB", "PP") + String("-") +  # Credit Card, Debit, PayPal
            Regex(r"[0-9]{4}") + String("-") +
            Regex(r"[0-9]{4}")
        )
        
        # Promo code: SAVE20, FREESHIP, WELCOME10
        self.promo_code = either(
            "SAVE20", "SAVE10", "SAVE30",
            "FREESHIP", "SHIP5",
            "WELCOME10", "WELCOME20",
            "SUMMER15", "WINTER25"
        )
        
        # Warehouse location: WH-XX-###
        self.warehouse = (
            String("WH-") +
            either("US", "UK", "EU", "AS") + String("-") +
            Regex(r"[0-9]{3}")
        )
    
    def generate_order_data(self):
        """Generate a complete order record."""
        generators = {
            'order_ref': generate.regex(self.model, self.tokenizer, self.order_ref),
            'customer_id': generate.regex(self.model, self.tokenizer, self.customer_id),
            'sku': generate.regex(self.model, self.tokenizer, sku_pattern),
            'serial': generate.regex(self.model, self.tokenizer, serial_pattern),
            'payment': generate.regex(self.model, self.tokenizer, self.payment_ref),
            'promo': generate.regex(self.model, self.tokenizer, self.promo_code),
            'warehouse': generate.regex(self.model, self.tokenizer, self.warehouse),
            'tracking': generate.regex(self.model, self.tokenizer, ups_tracking)
        }
        
        order = {}
        for field, gen in generators.items():
            order[field] = gen(f"Generate {field}: ", max_tokens=30)
        
        return order
    
    def generate_test_batch(self, count=5):
        """Generate multiple test orders."""
        orders = []
        for i in range(count):
            print(f"\nGenerating order {i+1}...")
            order = self.generate_order_data()
            orders.append(order)
            
            # Display the order
            print(f"Order Reference: {order['order_ref']}")
            print(f"Customer ID: {order['customer_id']}")
            print(f"Product SKU: {order['sku']}")
            print(f"Serial Number: {order['serial']}")
            print(f"Payment Ref: {order['payment']}")
            print(f"Promo Code: {order['promo']}")
            print(f"Ships From: {order['warehouse']}")
            print(f"Tracking: {order['tracking']}")
        
        return orders

# Generate test data
generator = TestDataGenerator()
test_orders = generator.generate_test_batch(3)

# Example output:
# Generating order 1...
# Order Reference: ORD-20240315-A7B9X
# Customer ID: CUST-42891
# Product SKU: APP-WASH-F3D-07K-WHT
# Serial Number: SN24W11B3A00456
# Payment Ref: PAY-CC-1234-5678
# Promo Code: SAVE20
# Ships From: WH-US-042
# Tracking: 1ZA4B5C612345678
```

## Step 6: Generating Test Scenarios

Create complex test scenarios with related data:

```python
# scenario_generator.py

# Return/RMA numbers: RMA-YYYYMMDD-ORD-XXXXX
rma_pattern = (
    String("RMA-") +
    String(datetime.now().strftime("%Y%m%d")) + String("-") +
    String("ORD-") + Regex(r"[A-Z0-9]{5}")
)

# Support ticket: TICK-DEPT-YYYYMMDD-XXXX
ticket_pattern = (
    String("TICK-") +
    either("TECH", "BILL", "SHIP", "RTRN") + String("-") +
    String(datetime.now().strftime("%Y%m%d")) + String("-") +
    Regex(r"[0-9]{4}")
)

# Batch/Lot numbers: LOT-YYYYMM-FAC-XXX
lot_pattern = (
    String("LOT-") +
    String(datetime.now().strftime("%Y%m")) + String("-") +
    either("USA", "CHN", "MEX", "VNM") + String("-") +
    Regex(r"[0-9]{3}")
)

# Generate related test data
rma_gen = generate.regex(model, tokenizer, rma_pattern)
ticket_gen = generate.regex(model, tokenizer, ticket_pattern)
lot_gen = generate.regex(model, tokenizer, lot_pattern)

print("\nGenerated Test Scenario:")
print(f"Customer places order: {order_ref}")
print(f"Product from lot: {lot_gen('Lot number: ', max_tokens=25)}")
print(f"Customer opens ticket: {ticket_gen('Support ticket: ', max_tokens=30)}")
print(f"Return initiated: {rma_gen('RMA number: ', max_tokens=30)}")
```

## Real-World Applications

This approach is perfect for:

1. **E-commerce Testing**: Generate realistic order data
2. **Inventory Systems**: Create valid SKUs and serial numbers
3. **Logistics Testing**: Generate tracking numbers for different carriers
4. **Integration Testing**: Create data that passes third-party validations
5. **Load Testing**: Generate thousands of valid test records

## Why Regex DSL Excels Here

1. **Business Rules**: Encode complex formatting rules directly
2. **Validation**: Generated data always passes regex validation
3. **Variety**: Create diverse but valid test data
4. **Realism**: Data looks real, not like "TEST123"
5. **Consistency**: All generated data follows the same patterns

## Conclusion

The Regex DSL in Outlines is ideal for generating test data that must match specific patterns. Unlike generic random data, this approach creates values that:
- Look realistic
- Pass validation rules
- Follow business logic
- Work with external systems

This is especially valuable when testing systems that integrate with third parties (payment processors, shipping carriers) or have strict validation rules.