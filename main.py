# Module level import
from recommendation import users_list, recommend
from data import getRandomUsers

# API imports
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel

# uvicorn main:app --reload


class TouristGroup(BaseModel):
    user_ids: list[str] = Body(...)


app = FastAPI()


@app.get("/")
async def root():
    return {
        "Select group of users from:": "/users",
        "Get itinerary prediction from:": "/predict",
    }


@app.post("/predict-for-group")
async def predict_for_group(tourist_group: TouristGroup):
    try:
        tourist_group, itinerary = recommend(tourist_group.user_ids)
        return {
            "Tourist Group": list(tourist_group),
            "Itinerary": list(itinerary)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


@app.get("/predict")
async def predict():
    try:
        tourist_group, itinerary = recommend(list(getRandomUsers()))
        return {
            "Tourist Group": list(tourist_group),
            "Itinerary": list(itinerary)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


@app.get("/users")
async def get_users():
    try:
        user_list = users_list()
        return {"users": list(user_list)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving users: {str(e)}")


# Run the application (for development purposes)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
