from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    ## In this value gets directly set by unpacking the json body of request by FastAPI


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