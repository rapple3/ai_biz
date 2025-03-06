import os
import json
from flask import Flask, request, jsonify
import openai
import tempfile
import faiss
import numpy as np
from datetime import datetime
import sys
import pkg_resources

# Initialize Flask app
app = Flask(__name__)

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create a temporary directory for files - with error handling
temp_dir = tempfile.gettempdir()
POLICIES_DIR = os.path.join(temp_dir, 'policies')
DATA_DIR = os.path.join(temp_dir, 'data')

# Create directories if they don't exist - with error handling
try:
    if not os.path.exists(POLICIES_DIR):
        os.makedirs(POLICIES_DIR)
except FileExistsError:
    pass  # Directory already exists, which is fine

try:
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
except FileExistsError:
    pass  # Directory already exists, which is fine

# Create sample policy files - with error handling
POLICIES = {
    "baggage_policy": "# SkyWay Airlines Baggage Policy\n\nAll passengers are allowed one carry-on bag and one personal item. Gold members get two free checked bags. Silver members get one free checked bag. Standard members must pay for all checked bags. Checked bags must not exceed 50 pounds (23 kg) and must not exceed 62 inches (157 cm) when adding length + width + height. Overweight or oversized bags will incur additional fees.",
    
    "cancellation_policy": "# SkyWay Airlines Cancellation Policy\n\nFull refunds are available for cancellations made 24 hours before departure. Cancellations made less than 24 hours before departure are eligible for a flight credit only. Gold members can cancel up to 2 hours before departure for a full refund. No-shows will not receive a refund or credit. Flight credits must be used within one year of issue.",
    
    "loyalty_program": "# SkyWay Airlines Loyalty Program\n\nSkyWay Airlines offers three loyalty tiers: Standard, Silver, and Gold. Silver status is achieved after 25,000 miles flown in a calendar year. Gold status is achieved after 50,000 miles flown in a calendar year. Silver members receive priority boarding, one free checked bag, and 25% bonus miles. Gold members receive priority boarding, two free checked bags, lounge access, and 50% bonus miles. Miles expire after 18 months of inactivity.",
    
    "flight_changes": "# SkyWay Airlines Flight Change Policy\n\nFlight changes can be made up to 2 hours before departure. Change fees apply based on fare class and loyalty tier. Gold members can change flights without fees. Silver members receive a 50% discount on change fees. Standard members pay full change fees. Same-day flight changes are available for a reduced fee of $75 for all members."
}

try:
    for policy_name, content in POLICIES.items():
        file_path = os.path.join(POLICIES_DIR, f"{policy_name}.txt")
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write(content)
except Exception as e:
    print(f"Error creating policy files: {e}")

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
try:
    with open(os.path.join(DATA_DIR, 'flights.json'), 'w') as f:
        json.dump(FLIGHTS_DATA, f)
    with open(os.path.join(DATA_DIR, 'customers.json'), 'w') as f:
        json.dump(CUSTOMERS_DATA, f)
except Exception as e:
    print(f"Error writing data files: {e}")

# Policy retriever using OpenAI embeddings and FAISS
class PolicyRetriever:
    def __init__(self, policies):
        self.policies = policies
        self.policy_texts = list(policies.values())
        self.policy_names = list(policies.keys())
        
        # Create embeddings for all policies
        self.embeddings = self._get_embeddings(self.policy_texts)
        
        # Create FAISS index
        self.dimension = len(self.embeddings[0])
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(self.embeddings).astype('float32'))
    
    def _get_embeddings(self, texts):
        """Get embeddings for a list of texts using OpenAI API"""
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * 1536 for _ in texts]  # 1536 is the dimension of text-embedding-3-small
    
    def get_relevant_policies(self, query, top_n=2):
        """Find the most relevant policies for a query"""
        # Get embedding for the query
        query_embedding = self._get_embeddings([query])[0]
        
        # Search the FAISS index
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), top_n)
        
        # Return the relevant policies
        results = []
        for i in I[0]:
            if i < len(self.policy_names):  # Safety check
                policy_name = self.policy_names[i]
                policy_text = self.policy_texts[i]
                results.append((policy_name, policy_text))
        
        return results
    
    def format_for_prompt(self, query):
        """Format relevant policies for inclusion in the prompt"""
        relevant_policies = self.get_relevant_policies(query)
        
        if not relevant_policies:
            return "No specific policy information found for this query."
        
        formatted_text = "Relevant SkyWay Airlines policies:\n\n"
        
        for policy_name, section in relevant_policies:
            formatted_text += f"From {policy_name.replace('_', ' ').title()} Policy:\n{section}\n\n"
            
        return formatted_text

# Initialize policy retriever
policy_retriever = PolicyRetriever(POLICIES)

# Function to get flight status
def get_flight_status(flight_id):
    for flight in FLIGHTS_DATA:
        if flight["flight_id"] == flight_id:
            return flight
    return None

# Function to get customer details
def get_customer_details(customer_id):
    for customer in CUSTOMERS_DATA:
        if customer["customer_id"] == customer_id:
            customer_data = customer.copy()
            flight_data = get_flight_status(customer_data["flight_id"])
            if flight_data:
                customer_data["flight"] = flight_data
            return customer_data
    return None

# Process chat messages
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
    if customer_details and "flight" in customer_details:
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
        # Get response from OpenAI using the new API format
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        
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
            
            # Use the new API format for the summary response
            summary_response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=300
            )
            
            structured_summary = summary_response.choices[0].message.content
            
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

# HTML content directly in the code instead of using templates
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Airline Customer Service Chatbot</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
        }
        
        body {
            background-color: #f5f5f5;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .chat-container {
            max-width: 600px;
            width: 100%;
            margin: 0 auto;
            background-color: white;
            height: 100vh;
            display: flex;
            flex-direction: column;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        
        .chat-header {
            background-color: #4285f4;
            color: white;
            padding: 15px;
            display: flex;
            align-items: center;
            position: relative;
        }
        
        .back-button {
            color: white;
            font-size: 24px;
            margin-right: 15px;
            cursor: pointer;
            text-decoration: none;
        }
        
        .header-title {
            flex-grow: 1;
            font-size: 18px;
            font-weight: 500;
        }
        
        .profile-button {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background-color: #c2dbff;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        
        .profile-icon {
            color: #4285f4;
            font-size: 24px;
        }
        
        .chat-messages {
            flex-grow: 1;
            padding: 15px;
            overflow-y: auto;
            background-color: #f5f5f5;
        }
        
        .message {
            margin-bottom: 15px;
            max-width: 80%;
            position: relative;
        }
        
        .sender-name {
            font-size: 12px;
            margin-bottom: 4px;
            color: #666;
            font-weight: 500;
        }
        
        .user-message {
            margin-left: auto;
            background-color: #4285f4;
            color: white;
            padding: 12px 16px;
            border-radius: 18px 18px 4px 18px;
        }
        
        .bot-message {
            margin-right: auto;
            background-color: #e5e5ea;
            color: #333;
            padding: 12px 16px;
            border-radius: 18px 18px 18px 4px;
        }
        
        .typing-indicator {
            padding: 12px 16px;
            border-radius: 18px;
            display: inline-block;
        }
        
        .loading-dots::after {
            content: '...';
            animation: loading 1.5s infinite;
        }
        
        @keyframes loading {
            0% { content: '.'; }
            33% { content: '..'; }
            66% { content: '...'; }
        }
        
        .chat-input {
            display: flex;
            align-items: center;
            padding: 10px 15px;
            background-color: white;
            border-top: 1px solid #e5e5ea;
        }
        
        .add-button {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background-color: #f5f5f5;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 10px;
            cursor: pointer;
            font-size: 20px;
            color: #4285f4;
        }
        
        .message-input {
            flex-grow: 1;
            border: none;
            background-color: #f5f5f5;
            border-radius: 20px;
            padding: 10px 15px;
            font-size: 16px;
            outline: none;
        }
        
        .send-button {
            background: none;
            border: none;
            color: #4285f4;
            font-weight: bold;
            font-size: 16px;
            padding: 0 10px;
            cursor: pointer;
        }
        
        .profile-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
            z-index: 100;
            align-items: center;
            justify-content: center;
        }
        
        .modal-content {
            background-color: white;
            border-radius: 12px;
            width: 90%;
            max-width: 400px;
            max-height: 80vh;
            overflow-y: auto;
            padding: 20px;
        }
        
        .modal-header {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .customer-list {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .customer-item {
            display: flex;
            align-items: center;
            padding: 10px;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .customer-item:hover {
            background-color: #f5f5f5;
        }
        
        .customer-avatar {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background-color: #4285f4;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 15px;
        }
        
        .gold-tier {
            background-color: #ffd700;
            color: #333;
        }
        
        .silver-tier {
            background-color: #c0c0c0;
            color: #333;
        }
        
        .standard-tier {
            background-color: #4285f4;
        }
        
        .customer-info {
            flex-grow: 1;
        }
        
        .customer-name {
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .customer-details {
            font-size: 12px;
            color: #666;
        }
        
        .escalation-panel {
            display: none;
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: white;
            padding: 20px;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
            z-index: 50;
            border-top-left-radius: 15px;
            border-top-right-radius: 15px;
        }
        
        .escalation-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .escalation-title {
            font-weight: 600;
            font-size: 18px;
        }
        
        .close-button {
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
        }
        
        .escalation-content {
            max-height: 50vh;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <a href="#" class="back-button">&#8249;</a>
            <div class="header-title">SkyWay Airlines Support</div>
            <div class="profile-button" id="profileButton">
                <div class="profile-icon">👤</div>
            </div>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="message bot-message">
                <div class="sender-name">SkyWay Assistant</div>
                Hello! I'm your SkyWay Airlines virtual assistant. How can I help you today?
            </div>
        </div>
        
        <div class="chat-input">
            <div class="add-button">+</div>
            <input type="text" class="message-input" id="userInput" placeholder="Type a message...">
            <button class="send-button" id="sendButton">Send</button>
        </div>
    </div>
    
    <div class="profile-modal" id="profileModal">
        <div class="modal-content">
            <div class="modal-header">Select Customer Profile</div>
            <div class="customer-list" id="customerList">
                <!-- Customer items will be added here by JavaScript -->
            </div>
        </div>
    </div>
    
    <div class="escalation-panel" id="escalationPanel">
        <div class="escalation-header">
            <div class="escalation-title">Agent Handoff Summary</div>
            <button class="close-button" id="closeEscalation">&times;</button>
        </div>
        <div class="escalation-content" id="escalationContent"></div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // DOM elements
            const chatMessages = document.getElementById('chatMessages');
            const userInput = document.getElementById('userInput');
            const sendButton = document.getElementById('sendButton');
            const profileButton = document.getElementById('profileButton');
            const profileModal = document.getElementById('profileModal');
            const customerList = document.getElementById('customerList');
            const escalationPanel = document.getElementById('escalationPanel');
            const closeEscalation = document.getElementById('closeEscalation');
            const escalationContent = document.getElementById('escalationContent');
            
            // Chat history
            let chatHistory = [];
            
            // Selected customer ID
            let selectedCustomerId = null;
            
            // Customer data
            const customers = [
                {
                    id: "C001",
                    name: "Jane Doe",
                    email: "jane@example.com",
                    tier: "Gold",
                    flight: "FL001 (NYC to LAX)",
                    status: "On Time"
                },
                {
                    id: "C002",
                    name: "John Smith",
                    email: "john@example.com",
                    tier: "Silver",
                    flight: "FL002 (LAX to CHI)",
                    status: "Delayed"
                },
                {
                    id: "C003",
                    name: "Alice Brown",
                    email: "alice@example.com",
                    tier: "Standard",
                    flight: "FL003 (MIA to DFW)",
                    status: "Cancelled"
                }
            ];
            
            // Initialize customer list
            function initializeCustomerList() {
                customerList.innerHTML = '';
                
                customers.forEach(customer => {
                    const customerItem = document.createElement('div');
                    customerItem.classList.add('customer-item');
                    customerItem.dataset.customerId = customer.id;
                    
                    const initials = customer.name.split(' ').map(n => n[0]).join('');
                    const tierClass = customer.tier.toLowerCase() + '-tier';
                    
                    customerItem.innerHTML = `
                        <div class="customer-avatar ${tierClass}">${initials}</div>
                        <div class="customer-info">
                            <div class="customer-name">${customer.name}</div>
                            <div class="customer-details">
                                ${customer.tier} Member • ${customer.flight} • ${customer.status}
                            </div>
                        </div>
                    `;
                    
                    customerItem.addEventListener('click', function() {
                        selectCustomer(customer);
                        profileModal.style.display = 'none';
                    });
                    
                    customerList.appendChild(customerItem);
                });
            }
            
            // Select customer
            function selectCustomer(customer) {
                selectedCustomerId = customer.id;
                
                // Update profile button
                const initials = customer.name.split(' ').map(n => n[0]).join('');
                const tierClass = customer.tier.toLowerCase() + '-tier';
                profileButton.innerHTML = `<div class="customer-avatar ${tierClass}">${initials}</div>`;
                
                // Add system message
                addSystemMessage(`Switched to ${customer.name}'s account`);
                
                // Clear chat history
                chatHistory = [];
                
                // Add welcome message
                const welcomeMessage = `Hello ${customer.name.split(' ')[0]}! I'm your SkyWay Airlines virtual assistant. How can I help you today?`;
                addMessage(welcomeMessage, 'bot');
                chatHistory.push({role: 'assistant', content: welcomeMessage});
            }
            
            // Initialize
            initializeCustomerList();
            
            // Event listeners
            profileButton.addEventListener('click', function() {
                profileModal.style.display = 'flex';
            });
            
            profileModal.addEventListener('click', function(e) {
                if (e.target === profileModal) {
                    profileModal.style.display = 'none';
                }
            });
            
            closeEscalation.addEventListener('click', function() {
                escalationPanel.style.display = 'none';
            });
            
            sendButton.addEventListener('click', sendMessage);
            
            userInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            function sendMessage() {
                const message = userInput.value.trim();
                if (message) {
                    // Add user message to chat
                    addMessage(message, 'user');
                    userInput.value = '';
                    
                    // Add to chat history
                    chatHistory.push({role: 'user', content: message});
                    
                    // Call the backend API
                    callChatAPI(message);
                }
            }
            
            function callChatAPI(userMessage) {
                // Show typing indicator
                const typingIndicator = document.createElement('div');
                typingIndicator.classList.add('message', 'bot-message', 'typing-indicator');
                typingIndicator.innerHTML = `
                    <div class="sender-name">SkyWay Assistant</div>
                    Typing<span class="loading-dots"></span>
                `;
                chatMessages.appendChild(typingIndicator);
                chatMessages.scrollTop = chatMessages.scrollHeight;
                
                // Prepare the request data
                const requestData = {
                    customer_id: selectedCustomerId,
                    message: userMessage,
                    chat_history: chatHistory
                };
                
                // Call the backend API
                fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestData)
                })
                .then(response => response.json())
                .then(data => {
                    // Remove typing indicator
                    chatMessages.removeChild(typingIndicator);
                    
                    // Add bot response to chat
                    addMessage(data.response, 'bot');
                    
                    // Add to chat history
                    chatHistory.push({role: 'assistant', content: data.response});
                    
                    // Handle escalation if needed
                    if (data.needs_escalation) {
                        escalationContent.innerHTML = data.structured_summary;
                        escalationPanel.style.display = 'block';
                    } else {
                        escalationPanel.style.display = 'none';
                    }
                })
                .catch(error => {
                    // Remove typing indicator
                    chatMessages.removeChild(typingIndicator);
                    
                    // Show error message
                    addMessage("Sorry, I'm having trouble connecting right now. Please try again later.", 'bot');
                    console.error('Error:', error);
                });
            }
            
            function addMessage(message, sender) {
                const messageElement = document.createElement('div');
                messageElement.classList.add('message');
                
                if (sender === 'user') {
                    messageElement.classList.add('user-message');
                    messageElement.textContent = message;
                } else {
                    messageElement.classList.add('bot-message');
                    messageElement.innerHTML = `
                        <div class="sender-name">SkyWay Assistant</div>
                        ${message}
                    `;
                }
                
                chatMessages.appendChild(messageElement);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            function addSystemMessage(message) {
                const messageElement = document.createElement('div');
                messageElement.style.textAlign = 'center';
                messageElement.style.margin = '10px 0';
                messageElement.style.fontSize = '12px';
                messageElement.style.color = '#666';
                messageElement.textContent = message;
                
                chatMessages.appendChild(messageElement);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        });
    </script>
</body>
</html>
"""

# Default route to serve the HTML
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return HTML_CONTENT

# API route for chat
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    customer_id = data.get('customer_id')
    user_message = data.get('message')
    chat_history = data.get('chat_history', [])
    
    result = process_chat(customer_id, user_message, chat_history)
    
    return jsonify(result)

# For local development
if __name__ == '__main__':
    app.run(debug=True)

@app.route('/debug/size', methods=['GET'])
def debug_size():
    # Get sizes of loaded modules
    module_sizes = {}
    for name, module in sys.modules.items():
        if hasattr(module, "__file__") and module.__file__:
            try:
                module_sizes[name] = os.path.getsize(module.__file__)
            except (OSError, AttributeError):
                module_sizes[name] = 0
    
    # Sort by size (largest first)
    sorted_modules = sorted(module_sizes.items(), key=lambda x: x[1], reverse=True)
    
    # Get installed package sizes
    package_sizes = {}
    for package in pkg_resources.working_set:
        try:
            package_path = os.path.dirname(package.location)
            package_size = 0
            for dirpath, dirnames, filenames in os.walk(package.location):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    package_size += os.path.getsize(file_path)
            package_sizes[package.key] = package_size
        except Exception:
            package_sizes[package.key] = 0
    
    # Sort packages by size
    sorted_packages = sorted(package_sizes.items(), key=lambda x: x[1], reverse=True)
    
    # Convert to human-readable format
    def human_size(bytes):
        units = ['B', 'KB', 'MB', 'GB']
        unit_index = 0
        while bytes >= 1024 and unit_index < len(units) - 1:
            bytes /= 1024
            unit_index += 1
        return f"{bytes:.2f} {units[unit_index]}"
    
    readable_modules = [(name, human_size(size)) for name, size in sorted_modules[:50]]  # Top 50 modules
    readable_packages = [(name, human_size(size)) for name, size in sorted_packages]
    
    return jsonify({
        "top_modules": readable_modules,
        "packages": readable_packages
    })