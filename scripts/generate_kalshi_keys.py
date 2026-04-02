# Created by Oliver Meihls

import os
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

def generate_kalshi_keys():
    # Make sure keys directory exists
    os.makedirs('keys', exist_ok=True)
    
    # Generate private key
    print("Generating RSA key pair for Kalshi...")
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096,
    )
    
    # Generate public key
    public_key = private_key.public_key()
    
    # Save private key to keys/kalshi_private.pem
    private_path = 'keys/kalshi_private.pem'
    with open(private_path, 'wb') as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
        
    # Save public key to keys/kalshi_public.pem
    public_path = 'keys/kalshi_public.pem'
    with open(public_path, 'wb') as f:
        f.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))
        
    print(f"Success! Keys saved to:\n- Private Key: {private_path}\n- Public  Key: {public_path}")
    print("\n!!! NEXT STEPS !!!")
    print("1. Go to the Kalshi website (https://kalshi.com) and log in.")
    print("2. Go to Settings -> Developer -> API.")
    print("3. Click 'Upload Key' or similar.")
    print("4. Tell Kalshi the contents of 'keys/kalshi_public.pem'.")
    print("5. Kalshi will give you a 'Key ID'. Copy that string.")
    print("6. Paste the Key ID into 'secrets/secrets.yaml' under kalshi: key_id: \"...\"")

if __name__ == "__main__":
    generate_kalshi_keys()
