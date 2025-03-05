from flask import Flask, request, jsonify, render_template
import os
import json
import pandas as pd
from datetime import datetime
import openai
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create a temporary directory for files
temp_dir = tempfile.gettempdir()
POLICIES_DIR = os.path.join(temp_dir, 'policies')
DATA_DIR = os.path.join(temp_dir, 'data')

if not os.path.exists(POLICIES_DIR):
    os.makedirs(POLICIES_DIR)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
# Create sample policy files
baggage_policy = "# SkyWay Airlines Baggage Policy\n\nAll passengers are allowed one carry-on bag and one personal item. Gold members get two free checked bags."
cancellation_policy = "# SkyWay Airlines Cancellation Policy\n\nFull refunds are available for cancellations made 24 hours before departure."

with open(os.path.join(POLICIES_DIR, 'baggage_policy.txt'), 'w') as f:
    f.write(baggage_policy)
with open(os.path.join(POLICIES_DIR, 'cancellation_policy.txt'), 'w') as f:
    f.write(cancellation_policy)
    
# Sample data
FLIGHTS_DATA = [
    {"flight_id": "FL001", "origin": "NYC", "destination": "LAX", "departure": "2025-03-04 10:00", "status": "On Time"},
    {"flight_id": "FL002", "origin": "LAX", "destination": "CHI", "departure": "2025-03-04 12:30", "status": "Delayed"},
    {"flight_id": "FL003", "origin": "MIA", "destination": "DFW", "departure": "2025-03-04 15:45", "status": "Cancelled"}
]
CUSTOMERS_DATA = [
    {"customer_id": "C001", "name": "Jane Doe", "email": "jane@example.com", "flight_id": "FL001", "loyalty_tier": "Gold"},
    {"customer_id": "C002", "name": "John Smith", "email": "john@example.com", "flight_id": "FL002", "loyalty_tier": "Silver"},
    {"customer_id": "C003", "name": "Alice Brown", "email": "alice@example.com", "flight_id": "FL003", "loyalty_tier": "Standard"}
]

# Write to temp directory
with open(os.path.join(DATA_DIR, 'flights.json'), 'w') as f:
    json.dump(FLIGHTS_DATA, f)
with open(os.path.join(DATA_DIR, 'customers.json'), 'w') as f:
    json.dump(CUSTOMERS_DATA, f)

# Simple policy retriever using TF-IDF
class SimplePolicy:
    def __init__(self, policy_dir):
        self.policy_dir = policy_dir
        self.policies = {}
        self.load_policies()
        self.vectorizer = TfidfVectorizer()
        
        # Fit vectorizer on all policy content
        all_content = list(self.policies.values())
        if all_content:
            self.vectorizer.fit(all_content)
    
    def load_policies(self):
        for filename in os.listdir(self.policy_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.policy_dir, filename)
                with open(file_path, 'r') as f:
                    content = f.read()
                    policy_name = filename.replace('_', ' ').replace('.txt', '')
                    self.policies[policy_name] = content
    
    def get_relevant_policies(self, query, top_n=2):
        if not self.policies:
            return []
            
        # Transform the query
        query_vector = self.vectorizer.transform([query])
        
        results = []
        for policy_name, content in self.policies.items():
            # Transform content
            content_vector = self.vectorizer.transform([content])
            
            # Calculate similarity
            similarity = cosine_similarity(query_vector, content_vector)[0][0]
            
            if similarity > 0.1:  # Threshold for relevance
                results.append((policy_name, content, similarity))
        
        # Sort by relevance and return top_n
        results.sort(key=lambda x: x[2], reverse=True)
        return [(name, content) for name, content, _ in results[:top_n]]
    
    def format_for_prompt(self, query):
        relevant_policies = self.get_relevant_policies(query)
        
        if not relevant_policies:
            return "No specific policy information found for this query."
        
        formatted_text = "Relevant SkyWay Airlines policies:\n\n"
        
        for policy_name, section in relevant_policies:
            formatted_text += f"From {policy_name.title()} Policy:\n{section}\n\n"
            
        return formatted_text

# Initialize policy retriever
policy_retriever = SimplePolicy(POLICIES_DIR)

# Convert data to pandas DataFrames
FLIGHTS_DB = pd.DataFrame(FLIGHTS_DATA)
CUSTOMERS_DB = pd.DataFrame(CUSTOMERS_DATA)

# Function to get flight status
def get_flight_status(flight_id):
    flight = FLIGHTS_DB[FLIGHTS_DB["flight_id"] == flight_id]
    if flight.empty:
        return None
    return flight.iloc[0].to_dict()

# Function to get customer details
def get_customer_details(customer_id):
    customer = CUSTOMERS_DB[CUSTOMERS_DB["customer_id"] == customer_id]
    if customer.empty:
        return None
    customer_data = customer.iloc[0].to_dict()
    flight_data = get_flight_status(customer_data["flight_id"])
    return {**customer_data, "flight": flight_data}

# Function to process chat with AI
def process_chat(customer_id, user_message, chat_history):
    # Get customer details for personalization
    customer_details = get_customer_details(customer_id) if customer_id else None
    
    # Get relevant policy information based on user message
    policy_info = policy_retriever.format_for_prompt(user_message)
    
    # Prepare system prompt
    system_prompt = """
    You are an airline customer service chatbot for SkyWay Airlines. Your role is to assist customers with 
    flight inquiries, booking issues, and general travel questions.
    
    Be helpful, concise, and friendly. If you cannot resolve an issue, prepare a 
    structured summary for a human agent by including "ESCALATE" in your response.
    
    When answering questions about policies, use the specific policy information provided.
    If no policy information is provided for a specific question, give a general answer
    and suggest the customer check the official website for detailed information.
    
    For flight status inquiries, provide the exact status and any relevant details like gate changes.
    For loyalty program questions, explain benefits based on the customer's tier when available.
    """
    
    # Combine system prompt with policy information
    if policy_info:
        system_prompt += f"\n\nReference the following policy information in your responses when relevant:\n\n{policy_info}"
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add customer context if available
    if customer_details:
        context = f"""
        Customer Information:
        - Name: {customer_details['name']}
        - Email: {customer_details['email']}
        - Loyalty Tier: {customer_details['loyalty_tier']}
        - Flight: {customer_details['flight']['flight_id']} ({customer_details['flight']['origin']} to {customer_details['flight']['destination']})
        - Departure: {customer_details['flight']['departure']}
        - Status: {customer_details['flight']['status']}
        
        When responding, personalize your answers using the customer's name and loyalty tier.
        For flight-related questions, reference their specific flight details.
        """
        messages.append({"role": "system", "content": context})
    
    # Add chat history (limited to last 5 messages)
    recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
    for message in recent_history:
        messages.append({"role": message["role"], "content": message["content"]})
    
    # Add current user message
    messages.append({"role": "user", "content": user_message})
    
    try:
        # Get response from OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message['content']
        
        # Check if the issue needs escalation
        if "ESCALATE" in ai_response:
            # Generate a structured summary for the agent
            summary_prompt = f"""
            Generate a structured summary for a human agent based on the following conversation:
            {json.dumps(chat_history)}
            User's last message: {user_message}
            
            Format:
            - Customer ID: {customer_id if customer_id else "Unknown"}
            - Problem Summary:
            - Attempted Solutions:
            - Recommended Next Steps:
            """
            
            summary_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=300
            )
            
            structured_summary = summary_response.choices[0].message['content']
            
            return {
                "response": ai_response.replace("ESCALATE", ""),
                "needs_escalation": True,
                "structured_summary": structured_summary
            }
        
        return {
            "response": ai_response,
            "needs_escalation": False
        }
    
    except Exception as e:
        print(f"Error: {e}")
        return {
            "response": "I'm having trouble processing your request. Please try again later.",
            "needs_escalation": True,
            "structured_summary": f"System error occurred: {str(e)}"
        }

# API route for chat
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    customer_id = data.get('customer_id')
    user_message = data.get('message')
    chat_history = data.get('chat_history', [])
    
    result = process_chat(customer_id, user_message, chat_history)
    
    return jsonify(result)

# Default route to serve the HTML
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return render_template('index.html')

# For local development
if __name__ == '__main__':
    app.run(debug=True) 