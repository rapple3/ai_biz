import os
import json
from flask import Flask, request, jsonify, render_template
import openai
import tempfile
from datetime import datetime
import sys
import pkg_resources

# Initialize Flask app with correct template folder path
# For Vercel deployment, we need to use absolute paths
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
app = Flask(__name__, template_folder=template_dir)

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

# Default route to serve the HTML
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return render_template('index.html')

# API route for chat
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    customer_id = data.get('customer_id')
    user_message = data.get('message')
    chat_history = data.get('chat_history', [])
    
    result = process_chat(customer_id, user_message, chat_history)
    
    return jsonify(result)

# Debug endpoint to check sizes
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

# For local development
if __name__ == '__main__':
    app.run(debug=True)