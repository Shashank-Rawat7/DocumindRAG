from pydantic import BaseModel, EmailStr, Field

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)
    ## In this value gets directly set by unpacking the json body of request by FastAPI

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: EmailStr

    class Config:
        from_attributes = True

    #In this we are using Config cuz we are getting data from SQLAlchemy Object here so we need
    # to set from_attributes = True so that it can read the data from SQLAlchemy Object


class Token(BaseModel):
    access_token: str
    token_type: str