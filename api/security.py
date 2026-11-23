import os
import jwt
from ninja.security import HttpBearer
from ninja.errors import HttpError
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"


class JWTAuth(HttpBearer):
    def authenticate(self, request, token):
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.PyJWTError:
            return None


class FarmerJWTAuth(HttpBearer):
    def authenticate(self, request, token):
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("role") == "farmer":
                return payload
        except jwt.PyJWTError:
            return None
        return None


class AdminJWTAuth(HttpBearer):
    def authenticate(self, request, token):
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("role") == "admin":
                return payload
        except jwt.PyJWTError:
            return None
        return None


jwt_auth = JWTAuth()
farmer_jwt_auth = FarmerJWTAuth()
admin_jwt_auth = AdminJWTAuth()

