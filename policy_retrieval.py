import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class PolicyRetriever:
    def __init__(self, policy_dir='policies'):
        self.policy_dir = policy_dir
        self.policies = {}
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.load_policies()
        self.fit_vectorizer()
        
    def load_policies(self):
        """Load all policy documents from the policy directory."""
        for filename in os.listdir(self.policy_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.policy_dir, filename)
                with open(file_path, 'r') as f:
                    content = f.read()
                    policy_name = filename.replace('_', ' ').replace('.txt', '')
                    self.policies[policy_name] = content
    
    def fit_vectorizer(self):
        """Fit the TF-IDF vectorizer on all policy documents."""
        self.vectorizer.fit(self.policies.values())
        
    def split_into_chunks(self, text, chunk_size=200):
        """Split text into chunks of approximately chunk_size words."""
        words = text.split()
        return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    def get_relevant_policies(self, query, top_n=3):
        """
        Retrieve the most relevant policy sections based on the query.
        Returns a list of (policy_name, relevant_section) tuples.
        """
        # Transform the query
        query_vector = self.vectorizer.transform([query])
        
        # Add common policy keywords to improve matching
        expanded_query = query.lower()
        if 'baggage' in expanded_query or 'luggage' in expanded_query or 'bag' in expanded_query:
            expanded_query += ' baggage allowance checked carry-on'
        elif 'cancel' in expanded_query or 'refund' in expanded_query:
            expanded_query += ' cancellation policy refund ticket'
        elif 'change' in expanded_query or 'rebook' in expanded_query or 'reschedule' in expanded_query:
            expanded_query += ' rebooking change flight reschedule'
        elif 'assistance' in expanded_query or 'wheelchair' in expanded_query or 'disability' in expanded_query:
            expanded_query += ' special assistance disability wheelchair'
        elif 'miles' in expanded_query or 'points' in expanded_query or 'status' in expanded_query or 'tier' in expanded_query:
            expanded_query += ' loyalty program miles points'
        
        # Use the expanded query if it's different from the original
        if expanded_query != query.lower():
            query_vector = self.vectorizer.transform([expanded_query])
            print(f"Expanded query: {expanded_query}")
        
        results = []
        
        # For each policy document
        for policy_name, content in self.policies.items():
            # Split into manageable chunks
            chunks = self.split_into_chunks(content)
            
            # Transform chunks
            chunk_vectors = self.vectorizer.transform(chunks)
            
            # Calculate similarity
            similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
            
            # Get the most relevant chunk
            if len(chunks) > 0:
                best_chunk_idx = similarities.argmax()
                best_chunk = chunks[best_chunk_idx]
                similarity_score = similarities[best_chunk_idx]
                
                # Lower the threshold to include more potentially relevant content
                if similarity_score > 0.05:
                    results.append((policy_name, best_chunk, similarity_score))
                    print(f"Match found in {policy_name}: score {similarity_score}")
        
        # Sort by relevance and return top_n
        results.sort(key=lambda x: x[2], reverse=True)
        return [(name, chunk) for name, chunk, _ in results[:top_n]]
    
    def format_for_prompt(self, query):
        """Format relevant policy information for inclusion in an AI prompt."""
        relevant_policies = self.get_relevant_policies(query)
        
        if not relevant_policies:
            return "No specific policy information found for this query."
        
        formatted_text = "Relevant SkyWay Airlines policies:\n\n"
        
        for policy_name, section in relevant_policies:
            formatted_text += f"From {policy_name.title()} Policy:\n{section}\n\n"
            
        return formatted_text 