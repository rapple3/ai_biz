from flask import Flask, request, jsonify, render_template
import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from policy_retrieval_langchain import PolicyRetrieverLangChain
import tempfile

# Create a temporary directory for files if we're in a serverless environment
if not os.path.exists('data') or not os.access('data', os.W_OK):
    temp_dir = tempfile.gettempdir()
    POLICIES_DIR = os.path.join(temp_dir, 'policies')
    DATA_DIR = os.path.join(temp_dir, 'data')
    
    if not os.path.exists(POLICIES_DIR):
        os.makedirs(POLICIES_DIR)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    # Copy sample data to temp directory if needed
    if not os.path.exists(os.path.join(DATA_DIR, 'flights.json')):
        # Include your sample data directly in the code
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
else:
    POLICIES_DIR = 'policies'
    DATA_DIR = 'data'

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize policy retriever
policy_retriever = PolicyRetrieverLangChain()

# Load data from JSON files
def load_data():
    try:
        with open(os.path.join(DATA_DIR, 'flights.json'), 'r') as f:
            flights = json.load(f)
        
        with open(os.path.join(DATA_DIR, 'customers.json'), 'r') as f:
            customers = json.load(f)
        
        return flights, customers
    except FileNotFoundError:
        # If files don't exist yet, return empty lists
        return [], []

# Load data and convert to pandas DataFrames
FLIGHTS_DATA, CUSTOMERS_DATA = load_data()
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
    
    # Debug: Print policy info to console
    print(f"Policy info retrieved: {policy_info}")
    
    # Initialize LangChain chat model
    chat_model = ChatOpenAI(temperature=0.7)
    
    # Prepare system messages
    system_messages = [
        SystemMessage(content="""
        You are an airline customer service chatbot for SkyWay Airlines. Your role is to assist customers with 
        flight inquiries, booking issues, and general travel questions.
        
        Be helpful, concise, and friendly. If you cannot resolve an issue, prepare a 
        structured summary for a human agent by including "ESCALATE" in your response.
        
        When answering questions about policies, use the specific policy information provided.
        If no policy information is provided for a specific question, give a general answer
        and suggest the customer check the official website for detailed information.
        
        For flight status inquiries, provide the exact status and any relevant details like gate changes.
        For loyalty program questions, explain benefits based on the customer's tier when available.
        """)
    ]
    
    # Add policy information if available
    if policy_info:
        system_messages.append(SystemMessage(content=f"Reference the following policy information in your responses when relevant:\n\n{policy_info}"))
    
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
        system_messages.append(SystemMessage(content=context))
    
    # Prepare message history
    messages = system_messages.copy()
    
    # Add chat history (limited to last 5 messages to keep context manageable)
    recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
    for message in recent_history:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"]))
    
    # Add current user message
    messages.append(HumanMessage(content=user_message))
    
    # Debug: Print the message structure
    print(f"Number of messages: {len(messages)}")
    for i, msg in enumerate(messages):
        print(f"Message {i}: {msg.type} - {msg.content[:50]}...")
    
    try:
        # Get response from LangChain
        response = chat_model.invoke(messages)
        ai_response = response.content
        
        # Debug: Print the raw response
        print(f"Raw AI response: {ai_response}")
        
        # Check if the issue needs escalation
        if "ESCALATE" in ai_response:
            # Generate a structured summary for the agent
            escalation_messages = [
                SystemMessage(content=f"""
                Generate a structured summary for a human agent based on the following conversation:
                {json.dumps(chat_history)}
                User's last message: {user_message}
                
                Format:
                - Customer ID: {customer_id if customer_id else "Unknown"}
                - Problem Summary:
                - Attempted Solutions:
                - Recommended Next Steps:
                """)
            ]
            
            summary_response = chat_model.invoke(escalation_messages)
            structured_summary = summary_response.content
            
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

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    customer_id = data.get('customer_id')
    user_message = data.get('message')
    chat_history = data.get('chat_history', [])
    
    result = process_chat(customer_id, user_message, chat_history)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
else:
    # This is for Vercel serverless deployment
    # Vercel expects a variable named 'app' to be the WSGI application
    app = app