import imaplib
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import os

# Initialize Firestore
db = firestore.Client(project="agent-fy-494108")

def test_imap():
    # Fetch all tenants to find the one with this email
    docs = db.collection("tenants").stream()
    
    found = False
    for doc in docs:
        data = doc.to_dict()
        config = data.get("config", {})
        gmail_user = config.get("gmail_address", "")
        
        if gmail_user == "farazashraf210@gmail.com":
            found = True
            gmail_pass = config.get("gmail_app_password", "")
            print(f"Found tenant {doc.id} with email {gmail_user}")
            print(f"Password length: {len(gmail_pass)}")
            print(f"Password contains spaces: {' ' in gmail_pass}")
            
            # Test 1: Exact password from database
            print("\n--- Test 1: Using exact password from DB ---")
            try:
                mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)
                mail.login(gmail_user, gmail_pass)
                print("SUCCESS: Logged in with exact password!")
                mail.logout()
            except Exception as e:
                print(f"FAILED: {e}")
                
            # Test 2: Sanitized password
            print("\n--- Test 2: Using sanitized password (no spaces) ---")
            sanitized_pass = gmail_pass.replace(" ", "").strip()
            print(f"Sanitized password length: {len(sanitized_pass)}")
            try:
                mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)
                mail.login(gmail_user, sanitized_pass)
                print("SUCCESS: Logged in with sanitized password!")
                mail.logout()
            except Exception as e:
                print(f"FAILED: {e}")
                
    if not found:
        print("Could not find any tenant with gmail_address = farazashraf210@gmail.com")

if __name__ == "__main__":
    test_imap()
