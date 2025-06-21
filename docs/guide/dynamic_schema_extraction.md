# Building a Research Agent with Dynamic Schemas

In this guide, we'll build an agent that researches topics and adapts its output structure based on what it finds. This is a genuine use case for dynamic schemas - the agent doesn't know what information it will discover until it starts researching.

## What We'll Build

A research agent that:

- Takes a research query
- Discovers what types of information are available
- Builds a schema based on its findings
- Returns structured data that matches what it found

## Why Dynamic Schemas?

When researching, you don't know in advance what you'll find:

- Researching a person might yield: biography, achievements, controversies
- Researching a company might yield: financials, products, leadership
- Researching an event might yield: date, location, participants, outcomes
- Researching a concept might yield: definition, history, applications

The agent needs to adapt its output to what it actually discovers!

## Prerequisites

```shell
# Create project
mkdir research-agent
cd research-agent
uv init
uv add outlines transformers genson
```

## Step 1: Understanding Genson

Genson creates JSON schemas from examples. Let's see how:

```python
# understanding_genson.py
from genson import SchemaBuilder
import json

# Create a schema builder
builder = SchemaBuilder()

# Add an example product
builder.add_object({
    "product": "iPhone 15",
    "price": 999,
    "rating": 4.5
})

# Get the schema
schema = builder.to_schema()
print(json.dumps(schema, indent=2))
```

Output:
```json
{
  "type": "object",
  "properties": {
    "product": {"type": "string"},
    "price": {"type": "integer"},
    "rating": {"type": "number"}
  },
  "required": ["price", "product", "rating"]
}
```

The schema describes the structure Genson found in our example!

## Step 2: Building a Simple Research Agent

Let's start with a basic agent that researches a topic in phases:

```python
# research_agent_v1.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import outlines
from outlines import Template
import json

# Load a small model
model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Phase 1: Discover what information is available
# Template: prompts/discover_info.jinja2
discover_template = Template("""
Research the topic: {{ topic }}

What types of information can you find about this topic?
List the main categories of information available:
""")

# Phase 2: Extract the discovered information
# Template: prompts/extract_info.jinja2
extract_template = Template("""
Research the topic: {{ topic }}

Provide the following information:
{{ categories }}

Format as JSON with these exact fields.
""")

def research_topic(topic):
    # Phase 1: Discover what's available
    prompt = discover_template(topic=topic)
    generate_text = outlines.generate.text(model, tokenizer)
    discovery = generate_text(prompt, max_tokens=200)
    
    print(f"Discovered about '{topic}':")
    print(discovery)
    
    # For now, we'll manually structure this
    # (We'll improve this in the next step)
    
# Example usage
research_topic("The Eiffel Tower")
research_topic("Marie Curie")
research_topic("The Python programming language")
```

Different topics yield completely different types of information!

## Step 3: Adding Structured Discovery

Now let's make the agent discover information categories in a structured way:

```python
# research_agent_v2.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import outlines
from outlines import Template
from genson import SchemaBuilder
import json

# Load model
model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def discover_information_structure(topic):
    """Discover what information is available about a topic."""
    # Template: prompts/discover_structure.jinja2
    template = Template("""
Research: {{ topic }}

List the types of information you can find (comma-separated):
""")
    
    # Use regex to get clean, parseable output
    field_pattern = r"[a-z_]+(?:, [a-z_]+)*"
    generate_fields = outlines.generate.regex(model, tokenizer, field_pattern)
    
    prompt = template(topic=topic)
    fields_str = generate_fields(prompt, max_tokens=100)
    fields = [f.strip() for f in fields_str.split(',') if f.strip()]
    
    return fields

def research_with_structure(topic, fields):
    """Research a topic and extract specific fields."""
    # Build a flexible schema
    schema = {
        "type": "object",
        "properties": {
            field: {"type": ["string", "array", "object", "null"]} 
            for field in fields
        },
        "additionalProperties": False
    }
    
    # Template: prompts/structured_research.jinja2
    template = Template("""
Research: {{ topic }}

Extract the following information:
{{ fields }}

Provide detailed information for each category.
""")
    
    generate_json = outlines.generate.json(model, tokenizer, schema_object=schema)
    prompt = template(topic=topic, fields="\n".join(f"- {f}" for f in fields))
    
    return generate_json(prompt)

# Example: Research different topics
topics = [
    "The Eiffel Tower",
    "Marie Curie", 
    "Python programming language"
]

for topic in topics:
    print(f"\n🔍 Researching: {topic}")
    
    # Discover structure
    fields = discover_information_structure(topic)
    print(f"Found categories: {fields}")
    
    # Extract information
    data = research_with_structure(topic, fields)
    print(f"\nExtracted data:")
    print(json.dumps(data, indent=2))
```

The agent discovers different structures for different topics!

## Step 4: Building an Adaptive Research Agent

Let's create an agent that learns from multiple research sessions:

```python
# adaptive_research_agent.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import outlines
from outlines import Template
from genson import SchemaBuilder
import json

class AdaptiveResearchAgent:
    def __init__(self, model_name="microsoft/phi-2"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.schema_builder = SchemaBuilder()
        self.known_patterns = {}  # topic_type -> common fields
        
    def identify_topic_type(self, topic):
        """Identify what type of topic this is."""
        template = Template("""
Classify this topic into a category:
{{ topic }}

Categories: person, place, technology, event, concept, organization
Category:""")
        
        # Simple classification
        generate = outlines.generate.regex(self.model, self.tokenizer, r"[a-z]+")
        prompt = template(topic=topic)
        return generate(prompt, max_tokens=10)
    
    def discover_fields(self, topic, topic_type):
        """Discover relevant fields based on topic and type."""
        template = Template("""
Topic: {{ topic }} (Type: {{ topic_type }})

What information should we collect about this {{ topic_type }}?
List relevant fields (comma-separated):""")
        
        generate = outlines.generate.regex(self.model, self.tokenizer, r"[a-z_]+(?:, [a-z_]+)*")
        prompt = template(topic=topic, topic_type=topic_type)
        fields_str = generate(prompt, max_tokens=100)
        
        fields = [f.strip() for f in fields_str.split(',') if f.strip()]
        
        # Learn patterns for future use
        if topic_type not in self.known_patterns:
            self.known_patterns[topic_type] = set()
        self.known_patterns[topic_type].update(fields)
        
        return fields
    
    def research(self, topic):
        """Research a topic with adaptive schema."""
        # Identify topic type
        topic_type = self.identify_topic_type(topic)
        print(f"Identified as: {topic_type}")
        
        # Get fields (use known patterns + discover new ones)
        base_fields = list(self.known_patterns.get(topic_type, []))
        new_fields = self.discover_fields(topic, topic_type)
        all_fields = list(set(base_fields + new_fields))
        
        print(f"Research fields: {all_fields}")
        
        # Build schema
        schema = {
            "type": "object",
            "properties": {
                field: {"type": ["string", "number", "array", "object", "null"]}
                for field in all_fields
            }
        }
        
        # Research with schema
        template = Template("""
Research: {{ topic }}
Type: {{ topic_type }}

Provide information for these aspects:
{{ fields }}
""")
        
        generate_json = outlines.generate.json(self.model, self.tokenizer, schema_object=schema)
        prompt = template(
            topic=topic, 
            topic_type=topic_type,
            fields="\n".join(f"- {f}" for f in all_fields)
        )
        
        result = generate_json(prompt)
        
        # Update schema knowledge
        self.schema_builder.add_object(result)
        
        return result

# Example usage
agent = AdaptiveResearchAgent()

# Research various topics - the agent learns and adapts
topics = [
    "Albert Einstein",
    "Marie Curie",  # Similar to Einstein, will reuse person fields
    "The Louvre Museum",
    "The Eiffel Tower",  # Similar to Louvre, will reuse place fields
    "Quantum Computing",
    "Machine Learning"  # Similar to Quantum Computing, will reuse technology fields
]

for topic in topics:
    print(f"\n{'='*50}")
    print(f"🔍 Researching: {topic}")
    result = agent.research(topic)
    print("\nResult:")
    print(json.dumps(result, indent=2))

print(f"\n📚 Learned patterns:")
for topic_type, fields in agent.known_patterns.items():
    print(f"{topic_type}: {sorted(fields)}")
```

## Step 5: Why This Matters

Let's see why dynamic schemas are essential for research agents:

```python
# research_comparison.py

# Static schema approach (doesn't work well)
static_person_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "birth_date": {"type": "string"},
        "occupation": {"type": "string"},
        "achievements": {"type": "array"}
    },
    "required": ["name", "birth_date", "occupation", "achievements"]
}

# Problem: What if we research a modern tech CEO vs historical figure?
# - Modern CEO: funding_rounds, company_valuations, twitter_handle
# - Historical figure: era, historical_significance, legacy
# - Artist: art_style, famous_works, exhibitions
# The static schema can't adapt!

# Dynamic approach allows the agent to:
# 1. Discover what information exists
# 2. Create appropriate structure
# 3. Handle any type of person

# Real example output from our adaptive agent:
example_outputs = {
    "Albert Einstein": {
        "birth_date": "March 14, 1879",
        "death_date": "April 18, 1955",
        "nationality": "German-American",
        "field": "Theoretical Physics",
        "famous_theories": ["Relativity", "E=mc²"],
        "nobel_prize": "1921",
        "contributions": "Revolutionized understanding of space and time"
    },
    
    "Elon Musk": {
        "birth_date": "June 28, 1971",
        "companies": ["Tesla", "SpaceX", "Twitter"],
        "net_worth": "$219 billion",
        "innovations": ["Electric vehicles", "Reusable rockets"],
        "social_media": "@elonmusk",
        "controversies": ["Twitter acquisition", "SEC disputes"]
    },
    
    "Van Gogh": {
        "birth_date": "March 30, 1853", 
        "death_date": "July 29, 1890",
        "art_movement": "Post-Impressionism",
        "famous_works": ["Starry Night", "Sunflowers"],
        "paintings_sold_lifetime": 1,
        "legacy": "Most famous artist posthumously"
    }
}

# Each person has completely different relevant fields!
# The agent discovered and adapted to what was important for each.
```

## When to Use Dynamic Schemas

Dynamic schemas are essential when:

- You don't know what fields will be relevant until runtime
- Different instances require completely different structures  
- You're building agents that need to adapt to discoveries
- Static schemas would require too many optional fields

## Next Steps

Extend this pattern to build:

1. **Web Scraping Agents**: Adapt to different website structures
2. **Data Integration Tools**: Handle varying API responses
3. **Document Analyzers**: Extract from diverse document types
4. **Question-Answering Systems**: Structure answers based on what's found
5. **Knowledge Base Builders**: Organize information as it's discovered

## Conclusion

We've built a research agent that truly needs dynamic schemas because:

- It can't know in advance what information exists about a topic
- Different topics require completely different fields
- The agent learns and improves its patterns over time
- A static schema would be either too restrictive or too generic

This demonstrates the real power of combining Outlines' structured generation capabilities with dynamic schema generation - creating agents that adapt to what they discover.
