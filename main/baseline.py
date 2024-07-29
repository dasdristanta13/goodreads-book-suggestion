import networkx as nx
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. Create the Knowledge Graph

def create_knowledge_graph(df):
    G = nx.Graph()
    
    for _, row in df.iterrows():
        # Add book node with more attributes
        G.add_node(row['book_id'], type='book', title=row['title'], 
                   isbn=row['isbn'], isbn13=row['isbn13'],
                   average_rating=row['average_rating'],
                   ratings_count=row['ratings_count'],
                   text_reviews_count=row['text_reviews_count'],
                   num_pages=row['num_pages'],
                   is_ebook=row['is_ebook'],
                   description=row['description'],
                   language_code=row['language_code'],
                   country_code=row['country_code'],
                   image_url=row['image_url'],
                   link=row['link'],
                   url=row['url'],
                   embedding=None)
        
        # Add authors node and edge (handling multiple authors)
        authors = row['authors'].split(',') if isinstance(row['authors'], str) else [row['authors']]
        for author in authors:
            G.add_node(author.strip(), type='author')
            G.add_edge(row['book_id'], author.strip(), relation='written_by')
        
        # Add publisher node and edge
        if pd.notna(row['publisher']):
            G.add_node(row['publisher'], type='publisher')
            G.add_edge(row['book_id'], row['publisher'], relation='published_by')
        
        # Add publication year, month, day nodes and edges
        if pd.notna(row['publication_year']):
            G.add_node(row['publication_year'], type='year')
            G.add_edge(row['book_id'], row['publication_year'], relation='published_in_year')
        if pd.notna(row['publication_month']):
            G.add_node(f"{row['publication_year']}-{row['publication_month']:02d}", type='month')
            G.add_edge(row['book_id'], f"{row['publication_year']}-{row['publication_month']:02d}", relation='published_in_month')
        if pd.notna(row['publication_day']):
            G.add_node(f"{row['publication_year']}-{row['publication_month']:02d}-{row['publication_day']:02d}", type='day')
            G.add_edge(row['book_id'], f"{row['publication_year']}-{row['publication_month']:02d}-{row['publication_day']:02d}", relation='published_on')
        
        # Add format node and edge
        if pd.notna(row['format']):
            G.add_node(row['format'], type='format')
            G.add_edge(row['book_id'], row['format'], relation='available_in')
        
        # Add series node and edge
        if pd.notna(row['series']):
            G.add_node(row['series'], type='series')
            G.add_edge(row['book_id'], row['series'], relation='part_of_series')
        
        # Add popular shelves as genres
        if pd.notna(row['popular_shelves']):
            shelves = eval(row['popular_shelves'])  # Assuming it's stored as a string representation of a list
            for shelf in shelves[:5]:  # Limit to top 5 shelves
                G.add_node(shelf, type='genre')
                G.add_edge(row['book_id'], shelf, relation='categorized_as')
        
        # Add similar books edges
        if pd.notna(row['similar_books']):
            similar_books = eval(row['similar_books'])  # Assuming it's stored as a string representation of a list
            for similar_book in similar_books:
                G.add_edge(row['book_id'], similar_book, relation='similar_to')
    
    return G

# 2. Implement Graph-based Retrieval with Personalized PageRank

class GraphRetriever:
    def __init__(self, graph, embedding_model):
        self.graph = graph
        self.embedding_model = embedding_model
        self.node_embeddings = self._compute_node_embeddings()
    
    def _compute_node_embeddings(self):
        node_texts = {node: self._node_to_text(node) for node in self.graph.nodes()}
        embeddings = self.embedding_model.encode(list(node_texts.values()))
        return dict(zip(node_texts.keys(), embeddings))
    
    def _node_to_text(self, node):
        node_data = self.graph.nodes[node]
        if node_data['type'] == 'book':
            return f"Book: {node_data['title']}. Description: {node_data['description'][:200]}..."
        elif node_data['type'] == 'author':
            return f"Author: {node}"
        elif node_data['type'] == 'genre':
            return f"Genre: {node}"
        elif node_data['type'] == 'year':
            return f"Year: {node}"
        elif node_data['type'] == 'publisher':
            return f"Publisher: {node}"
        elif node_data['type'] == 'series':
            return f"Series: {node}"
        else:
            return str(node)
    
    def retrieve(self, query, user_id=None, top_k=5):
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, list(self.node_embeddings.values()))[0]
        
        if user_id:
            personalized = self._personalized_pagerank(user_id)
            combined_scores = 0.7 * similarities + 0.3 * personalized
        else:
            combined_scores = similarities
        
        top_indices = combined_scores.argsort()[-top_k:][::-1]
        return [list(self.node_embeddings.keys())[i] for i in top_indices]
    
    def _personalized_pagerank(self, user_id, alpha=0.85, max_iter=100):
        personalization = {node: 1 if node == user_id else 0 for node in self.graph.nodes()}
        pagerank = nx.pagerank(self.graph, alpha=alpha, personalization=personalization, max_iter=max_iter)
        return np.array([pagerank.get(node, 0) for node in self.node_embeddings.keys()])
    
    def expand_context(self, nodes, depth=2):
        context = set(nodes)
        for _ in range(depth):
            new_context = set()
            for node in context:
                new_context.update(self.graph.neighbors(node))
            context.update(new_context)
        return list(context)

# 3. Integrate with RAG for Recommendations

class BookRecommender:
    def __init__(self, graph, retriever, llm, tokenizer):
        self.graph = graph
        self.retriever = retriever
        self.llm = llm
        self.tokenizer = tokenizer
    
    def get_recommendations(self, user_query, user_id=None):
        relevant_nodes = self.retriever.retrieve(user_query, user_id)
        context_nodes = self.retriever.expand_context(relevant_nodes)
        context = self._prepare_context(context_nodes, user_id)
        
        prompt = f"""Context: {context}

User Query: {user_query}

Based on the provided context and user query, please recommend a book or a few books that best match the user's interests. For each recommendation, provide a brief explanation of why it's a good fit.

Recommendation:"""

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.llm.device)
        
        with torch.no_grad():
            output = self.llm.generate(input_ids, max_length=500, num_return_sequences=1, temperature=0.7)
        
        recommendation = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return recommendation
    
    def _prepare_context(self, nodes, user_id=None):
        context = []
        for node in nodes:
            node_data = self.graph.nodes[node]
            if node_data['type'] == 'book':
                context.append(f"Book: {node_data['title']}")
                context.append(f"Description: {node_data['description'][:200]}...")
                context.append(f"Average Rating: {node_data['average_rating']} ({node_data['ratings_count']} ratings)")
                context.append(f"Language: {node_data['language_code']}")
                authors = [n for n in self.graph.neighbors(node) if self.graph.nodes[n]['type'] == 'author']
                context.append(f"Authors: {', '.join(authors)}")
                genres = [n for n in self.graph.neighbors(node) if self.graph.nodes[n]['type'] == 'genre']
                context.append(f"Genres: {', '.join(genres[:5])}")
            elif node_data['type'] == 'author':
                books = [self.graph.nodes[n]['title'] for n in self.graph.neighbors(node) if self.graph.nodes[n]['type'] == 'book']
                context.append(f"Author: {node}, Books: {', '.join(books[:5])}")
        
        if user_id:
            user_context = self._get_user_context(user_id)
            context.append(f"User Context: {user_context}")
        
        return "\n".join(context)
    
    def _get_user_context(self, user_id):
        user_node = self.graph.nodes[user_id]
        read_books = [self.graph.nodes[n]['title'] for n in self.graph.neighbors(user_id) if self.graph.nodes[n]['type'] == 'book']
        favorite_genres = self._get_user_favorite_genres(user_id)
        favorite_authors = self._get_user_favorite_authors(user_id)
        return f"Read Books: {read_books[:5]}, Favorite Genres: {favorite_genres}, Favorite Authors: {favorite_authors}"
    
    def _get_user_favorite_genres(self, user_id):
        genre_counts = {}
        for book_id in self.graph.neighbors(user_id):
            if self.graph.nodes[book_id]['type'] == 'book':
                for genre in self.graph.neighbors(book_id):
                    if self.graph.nodes[genre]['type'] == 'genre':
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
        return sorted(genre_counts, key=genre_counts.get, reverse=True)[:3]
    
    def _get_user_favorite_authors(self, user_id):
        author_counts = {}
        for book_id in self.graph.neighbors(user_id):
            if self.graph.nodes[book_id]['type'] == 'book':
                for author in self.graph.neighbors(book_id):
                    if self.graph.nodes[author]['type'] == 'author':
                        author_counts[author] = author_counts.get(author, 0) + 1
        return sorted(author_counts, key=author_counts.get, reverse=True)[:3]
    
    def update_graph(self, user_id, book_id, rating):
        if not self.graph.has_edge(user_id, book_id):
            self.graph.add_edge(user_id, book_id, relation='rated', weight=rating)
        else:
            self.graph[user_id][book_id]['weight'] = rating
        
        # Update user embedding
        user_books = [self.graph.nodes[n]['title'] for n in self.graph.neighbors(user_id) if self.graph.nodes[n]['type'] == 'book']
        user_text = f"User who has read: {', '.join(user_books)}"
        user_embedding = self.retriever.embedding_model.encode([user_text])[0]
        self.graph.nodes[user_id]['embedding'] = user_embedding
        
        # Update book embedding
        book_text = self.retriever._node_to_text(book_id)
        book_embedding = self.retriever.embedding_model.encode([book_text])[0]
        self.graph.nodes[book_id]['embedding'] = book_embedding
        
        # Update retriever's node embeddings
        self.retriever.node_embeddings[user_id] = user_embedding
        self.retriever.node_embeddings[book_id] = book_embedding

# Usage example

# Load your data
df = pd.read_csv('book_data.csv')

# Create knowledge graph
G = create_knowledge_graph(df)

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize graph retriever
retriever = GraphRetriever(G, embedding_model)

# Initialize quantized Llama-2 model
model_name = "meta-llama/Llama-2-7b-chat-hf"  # or another Llama-2 variant
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Initialize recommender
recommender = BookRecommender(G, retriever, llm, tokenizer)

# Get recommendations
user_id = "user123"
user_query = "I'm looking for a science fiction book with complex characters, preferably written in the last decade."
recommendation = recommender.get_recommendations(user_query, user_id)
print(recommendation)

# Simulate user interaction and update graph
book_id = "book456"
rating = 5
recommender.update_graph(user_id, book_id, rating)