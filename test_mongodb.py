from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime

try:
    # Connect to MongoDB
    print("Attempting to connect to MongoDB...")
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
    
    # Test the connection
    client.admin.command('ping')
    print("✅ Successfully connected to MongoDB!")
    
    # Get or create the database and collection
    db = client['translation_db']
    collection = db['arabic_translator']
    
    # Insert a test document
    test_doc = {
        'original_text': 'Hello World',
        'translated_text': 'مرحبا بالعالم',
        'timestamp': datetime.utcnow(),
        'source_language': 'en',
        'target_language': 'ar'
    }
    
    # Insert the document
    result = collection.insert_one(test_doc)
    print(f"✅ Inserted test document with ID: {result.inserted_id}")
    
    # Count documents in the collection
    count = collection.count_documents({})
    print(f"📊 Total documents in 'arabic_translator' collection: {count}")
    
    # Show all documents
    print("\nDocuments in 'arabic_translator' collection:")
    for doc in collection.find():
        print(f"- ID: {doc['_id']}")
        print(f"  Original: {doc['original_text']}")
        print(f"  Translated: {doc['translated_text']}")
        print(f"  Timestamp: {doc['timestamp']}")
    
except ConnectionFailure as e:
    print("❌ MongoDB connection failed!")
    print(f"Error details: {e}")
    
except Exception as e:
    print(f"❌ An error occurred: {e}")
    
finally:
    if 'client' in locals():
        client.close()
        print("\nConnection closed.")
