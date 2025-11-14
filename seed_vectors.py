from dotenv import load_dotenv
load_dotenv()

from utils.mongoDB  import insert_vector_data
insert_vector_data("gift_coupons", r"D:\New folder\backend_price_airlines\ChatSB-Backend\master_offers.csv")  # Insert data into MongoDB Atlas if not already present