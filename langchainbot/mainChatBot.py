import os
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

# IMPORT ML MODEL
from models.hotel import find_best_hotel
from models.restaurant import find_best_restaurant
from models.visit_place import suggest_trip

# CONFIGURATION 
vector_db_path = r"D:\travel\rag_code\vector_data_set"
collection_name = "place_info_v1"

# SETUP ENGINES
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatOllama(model="gemma3:1b", temperature=0.1)

vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embedding,
    persist_directory=vector_db_path
)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.5}
)


def get_hotel_response(user_input):
    """Manual Logic to trigger the ML Model"""
    print("🤖 [ML Logic Activated]")
    budget = int(input("Enter the Budget For the Hotel: "))
    night = int(input("How many nights or day you want to stay: "))
    area = str(input("what is your preferred area to stay: "))
    room_type = str(input("What type of room you want? "))
    hotel_data = find_best_hotel(budget, night, area, room_type)

    if isinstance(hotel_data, pd.DataFrame) and not hotel_data.empty:
        response = "🏨 **Top Hotel Recommendations for You:**\n"
        for index, row in hotel_data.iterrows():
            response += (
                f"\n      **{row['Hotel Name']}**"
                f"\n   📍 Area: {row['Area Part']}"
                f"\n   💰 Price: ₹{row['Price (INR)']} per night"
                f"\n   ⭐ Rating: {row['Rating']}/5\n"
            )

        return response
    return "🏨 **Hotel Recommendation**: I'm sorry, I couldn't find a hotel matching your budget in our records."


def get_restaurant_response(user_input):
    print("🤖 [Restaurant Logic Activated]")
    try:
        budget = int(input("Enter the Budget For the restaurant: "))
        group_size = int(input("How many people you are: "))
        area = str(input("What is your preferred area: "))
        preferred_cuisine = str(input("What type of food (Cuisine): "))
    except ValueError:
        return "❌ Please enter valid numbers for budget and group size."

    restaurant_data = find_best_restaurant(budget, group_size, area, preferred_cuisine)

    if isinstance(restaurant_data, pd.DataFrame) and not restaurant_data.empty:
        response = "🍴 **Top Restaurant Recommendations for You:**\n"

        for index, row in restaurant_data.iterrows():
            # Check if 'Area Part' exists, otherwise use 'Area' or whatever is in your CSV
            # I am using .get() or checking names to prevent the KeyError
            location = row.get('Area Part') or row.get('Area') or row.get('Location') or "Jogeshwari"

            response += (
                f"\n      **{row['Restaurant Name']}**"
                f"\n   📍 Location: {location}"
                f"\n   🍱 Cuisine: {row.get('Cuisine', 'Normal')}"
                f"\n   💰 Price for Two: ₹{row['Price for Two (INR)']}"
                f"\n   ⭐ Rating: {row['Rating']}/5\n"
            )
        return response

    return "🍴 **Restaurant Search**: I'm sorry, I couldn't find any restaurants matching your criteria."


def get_visit_place_response(user_input):
    print("🤖 [Sightseeing Logic Activated]")

    try:
        # Collect inputs specifically for the trip
        budget = int(input("Enter your total budget for sightseeing: "))
        group_size = int(input("How many people are in your group: "))

        print("Choose transport mode: Auto, Private_Car, Public_Bus")
        transport_mode = input("Enter transport mode: ").strip()

        allowed_types = ["Heritage", "Nature", "Beach", "Shopping", "Entertainment", "Park", "Landmark", "Exhibition",
                         "Religious"]
        preferred_type = input("Enter type of place you preferred (example: beach, nature, etc)")
        if preferred_type not in allowed_types and preferred_type != "":
            print(f"⚠️ '{preferred_type}' is not a standard category. I will search all places for you.")
            p_type = ""  # Reset to empty to search everything
    except ValueError:
        return "❌ Please enter valid numbers for budget and group size."

    #  Calling suggest_trip ML Model
    trip_options = suggest_trip(budget, group_size, transport_mode, preferred_type)

    if isinstance(trip_options, pd.DataFrame) and not trip_options.empty:
        response = f"🗺️ **Top Sightseeing Recommendations for {group_size} People:**\n"

        # Use for loop to show all recommendations
        for index, row in trip_options.iterrows():
            response += (
                f"\n      📍 **{row['Place_Name']}**"
                f"\n   🏛️ Type: {row['Type']}"
                f"\n   ⭐ Rating: {row['Rating']}/5"
                f"\n   🚗 Transport Cost: ₹{row['Predicted_Transport_Cost']:.2f} ({transport_mode})"
                f"\n   🎟️ Entry Fee: ₹{row['Entry_Fee']} per person"
                f"\n   💰 **Total Estimated Trip Cost: ₹{row['Total_Trip_Cost']:.2f}**\n"
            )
        return response

    return "🗺️ **Trip Search**: I'm sorry, I couldn't find any places that fit within your budget for this group size."

# CHATBOT LOGIC 
chat_history = []


def chatbot_response(user_input):
    global chat_history

    # Since 1b models struggle with tools, we use keywords or a simple prompt
    hotel_keywords = ["hotel", "stay", "room", "accommodation", "where can i sleep"]
    restaurant_keyword = ["restaurant", "where i can eat food", "dinner", "lunch", "breakfast", "meal", "eat"]
    visit_keywords = ["visit", "place", "trip", "sightseeing", "caves", "temple", "tour"]
    if any(word in user_input.lower() for word in hotel_keywords):
        return get_hotel_response(user_input)

    if any(word in user_input.lower() for word in restaurant_keyword):
        return get_restaurant_response(user_input)


    if any(word in user_input.lower() for word in visit_keywords):
        return get_visit_place_response(user_input)

    # RAG Pipeline for History/Facts 
    if chat_history:
        context_q = f"History: {chat_history[-4:]}\nQuestion: {user_input}\nStandalone Question:"
        query = llm.invoke(context_q).content.strip()
    else:
        query = user_input

    # Retrieve from PDF
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Final prompt
    final_prompt = f"""
        SYSTEM: You are a factual travel guide. 
        1. Use ONLY the Context provided. 
        2. If the Context mentions multiple different places (e.g., Juhu and Aarey), ONLY answer about the place mentioned in the Question.
        3. Do NOT mix facts between different locations. 
        4. If the Context doesn't specifically link a fact to the requested place, do not include it.
        5. If you are unsure, say "I don't have enough specific info on that."

        Context: 
        {context}

        Question: {query}
        Answer:"""

    response = llm.invoke(final_prompt).content

    # Update History
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    return response


#  Main Loop
print("🤖: Hello! I'm your hybrid guide (ML + RAG). Type 'exit' to stop.")
while True:
    user_query = input("\n👤 You: ")
    if user_query.lower() in ["exit", "quit"]: break

    reply = chatbot_response(user_query)

    print(f"🤖 AI: {reply}")
