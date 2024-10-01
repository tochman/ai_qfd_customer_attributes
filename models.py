from typing import List
from pydantic import BaseModel, Field


class Statement(BaseModel):
    """Model representing a customer statement with its sentiment score."""

    statement: str = Field(description="The original customer statement")
    score: float = Field(description="Sentiment score between -1 and 1")


class SentimentResult(BaseModel):
    """Model representing the sentiment analysis result for a statement."""

    statement: str = Field(description="The original customer statement")
    score: float = Field(description="Sentiment score between -1 and 1")
    label: str = Field(description="Sentiment label: Positive, Neutral, or Negative")


class SentimentResults(BaseModel):
    """Model representing a list of sentiment analysis results."""

    results: List[SentimentResult] = Field(description="List of sentiment results")


class TertiaryAttribute(BaseModel):
    """Model representing a tertiary attribute."""

    attribute: str = Field(description="The name of the tertiary attribute")
    statements: List[Statement] = Field(
        description="List of customer statements related to this attribute along with their sentiment scores"
    )


class SecondaryAttribute(BaseModel):
    """Model representing a secondary attribute."""

    attribute: str = Field(description="The name of the secondary attribute")
    tertiary_attributes: List[TertiaryAttribute] = Field(
        description="List of tertiary attributes under this secondary attribute"
    )


class PrimaryAttribute(BaseModel):
    """Model representing a primary attribute."""

    primary_attribute: str = Field(description="The name of the primary attribute")
    secondary_attributes: List[SecondaryAttribute] = Field(
        description="List of secondary attributes under this primary attribute"
    )


class CustomerAttributes(BaseModel):
    """Model representing a list of customer attributes."""

    attributes: List[PrimaryAttribute] = Field(
        description="List of primary customer attributes with nested secondary and tertiary attributes"
    )


class BusinessAnalysis(BaseModel):
    """Model representing the business analysis based on previous results"""

    introduction: str = Field(
        description="The introduction of the narrative to the analysis"
    )
    analysis: str = Field(description="The main body of the narrative of the analysis")
    recommendations: str = Field(
        description="Summary and recommendations for the operations management team"
    )
    derived_attributes_introduction: str = Field(
        description="The introduction of the process of deriving customer attributes from survey statements"
    )
    conclusions: str = Field(description="Summary and conclusion of the analysis")
