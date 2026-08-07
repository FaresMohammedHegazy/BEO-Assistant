import os
from rag.vector_store import VectorStore

def populate_vector_db():
    db_path = os.path.join(os.path.dirname(__file__), 'aurelia.db')
    
    print("="*50)
    
    vs = VectorStore(store_path=db_path)
    vs.initialize()
    
    knowledge_base = [
        "The Grand Magnolia Ballroom (ROOM_101) has a strict maximum capacity of 300 guests due to fire code enforcement. Exceeding this limit is a severe safety violation.",
        "Eleanor Vance is a VIP guest. She has a severe nut allergy and requires a strict vegan diet. All meals prepared for her must avoid cross-contamination with nuts or animal products.",
        "Event EVT_999 is currently pending. It requires a mandatory deposit of $20,000.00 to confirm the booking and secure the room.",
        "Standard Conference Rooms have a maximum capacity of 50 people and are fully compliant with standard safety regulations.",
        "Safe ingredients for vegans include Quinoa, Roasted Chickpeas, and Tofu. Butter is not vegan. Almonds and Macadamia Oil contain nuts and must be avoided for guests with nut allergies."
    ]
    
    print("Adding texts using the native add_texts method...")
    vs.add_texts(knowledge_base)
    
    print("SUCCESS! The VectorStore has been populated correctly.")
    print("="*50)

if __name__ == "__main__":
    populate_vector_db()