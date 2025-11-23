from pydantic import BaseModel, Field


class QueryInput(BaseModel):
    """Input schema for the query tool."""

    query: str = Field(
        description=(
            "Natural language question about the user's portfolio data "
            "(e.g., 'What are my top 5 holdings by value?', "
            "'Show my total portfolio return')"
        )
    )
    user_id: str = Field(
        description=("User ID for filtering database queries to ensure " "data privacy")
    )
