from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
import tempfile

class PolicyRetrieverLangChain:
    def __init__(self, policy_dir=None):
        if policy_dir is None:
            if not os.path.exists('policies') or not os.access('policies', os.W_OK):
                temp_dir = tempfile.gettempdir()
                policy_dir = os.path.join(temp_dir, 'policies')
                
                if not os.path.exists(policy_dir):
                    os.makedirs(policy_dir)
                
                # Create sample policy files if they don't exist
                self.create_sample_policies(policy_dir)
            else:
                policy_dir = 'policies'
        
        self.policy_dir = policy_dir
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
        self.vector_store = None
        self.initialize_vector_store()
        
    def create_sample_policies(self, policy_dir):
        # Create some basic policy files for the serverless environment
        baggage_policy = "# SkyWay Airlines Baggage Policy\n\nAll passengers are allowed one carry-on bag and one personal item. Gold members get two free checked bags."
        cancellation_policy = "# SkyWay Airlines Cancellation Policy\n\nFull refunds are available for cancellations made 24 hours before departure."
        
        with open(os.path.join(policy_dir, 'baggage_policy.txt'), 'w') as f:
            f.write(baggage_policy)
        with open(os.path.join(policy_dir, 'cancellation_policy.txt'), 'w') as f:
            f.write(cancellation_policy)
        
    def load_policies(self):
        """Load all policy documents from the policy directory."""
        documents = []
        for filename in os.listdir(self.policy_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.policy_dir, filename)
                with open(file_path, 'r') as f:
                    content = f.read()
                    policy_name = filename.replace('_', ' ').replace('.txt', '')
                    doc = Document(
                        page_content=content,
                        metadata={"source": filename, "policy_name": policy_name}
                    )
                    documents.append(doc)
        return documents
    
    def initialize_vector_store(self):
        """Initialize the vector store with policy documents."""
        documents = self.load_policies()
        if not documents:
            print("Warning: No policy documents found.")
            return
            
        # Split documents into chunks
        splits = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        print(f"Vector store initialized with {len(splits)} document chunks")
    
    def get_relevant_policies(self, query, top_k=3):
        """Retrieve the most relevant policy sections based on the query."""
        if not self.vector_store:
            print("Vector store not initialized.")
            return []
            
        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(query, k=top_k)
        
        # Format results
        results = []
        for doc in docs:
            policy_name = doc.metadata.get("policy_name", "Unknown Policy")
            results.append((policy_name, doc.page_content))
            
        return results
    
    def format_for_prompt(self, query):
        """Format relevant policy information for inclusion in an AI prompt."""
        relevant_policies = self.get_relevant_policies(query)
        
        if not relevant_policies:
            return "No specific policy information found for this query."
        
        formatted_text = "Relevant SkyWay Airlines policies:\n\n"
        
        for policy_name, section in relevant_policies:
            formatted_text += f"From {policy_name.title()} Policy:\n{section}\n\n"
            
        return formatted_text 