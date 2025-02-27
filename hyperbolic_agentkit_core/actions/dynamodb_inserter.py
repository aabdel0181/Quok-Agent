from collections.abc import Callable
from decimal import Decimal
from pydantic import BaseModel, Field
import boto3
import os
from hyperbolic_agentkit_core.actions.hyperbolic_action import HyperbolicAction

INSERT_PROMPT = """
Insert JSON data into a DynamoDB table. This tool uses the ACCESS_KEY and SECRET_ACCESS_KEY
from the environment to connect to DynamoDB. If the table name is not provided, the tool will 
decide based on the provided data_type. For example, if data_type is 'GPUHealthCheck', the data
will be inserted into the 'GPUHealthChecks' table.
"""

class DynamoDBInsertInput(BaseModel):
    """
    Input schema for inserting data into DynamoDB.
    - data: The JSON data to insert into the table.
    - table: (Optional) Explicit DynamoDB table name to use.
    - data_type: The type of data being inserted. Used to decide the table if 'table' is not provided.
    """
    data: dict = Field(..., description="The JSON data to insert into DynamoDB")
    table: str | None = Field(None, description="Optional: Specify the table name. If omitted, the table is chosen based on data_type.")
    data_type: str = Field(..., description="The type of data (e.g., 'GPUHealthCheck'). Used to select the table if 'table' is not provided.")

def convert_floats_to_decimals(obj):
    """
    Recursively convert float values in the given object to Decimal.
    """
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: convert_floats_to_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_decimals(item) for item in obj]
    else:
        return obj

def execute_dynamodb_insert(data: dict, table: str | None, data_type: str) -> str:
    """
    Insert the provided data into a DynamoDB table after converting floats to Decimals.

    If no table is explicitly provided, choose one based on the data_type.
    Uses the ACCESS_KEY and SECRET_ACCESS_KEY from environment variables.
    """
    # Determine the target table if not provided
    if table is None:
        table_mapping = {
            "GPUHealthCheck": "GPUHealthChecks",
            # Add additional mappings as needed for different data types
        }
        table = table_mapping.get(data_type, "GeneralData")
    
    # Fetch AWS credentials from the environment
    ACCESS_KEY = os.getenv("ACCESS_KEY")
    SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")
    if not ACCESS_KEY or not SECRET_ACCESS_KEY:
        return "Error: ACCESS_KEY and/or SECRET_ACCESS_KEY environment variables are not set."
    
    try:
        # Convert all float types to Decimal
        data = convert_floats_to_decimals(data)
        
        # Initialize a DynamoDB resource using boto3
        dynamodb = boto3.resource(
            "dynamodb",
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_ACCESS_KEY,
            region_name="us-east-2"  # NOTE: make sure the region is correct (i found out the hard way)
        )
        dynamo_table = dynamodb.Table(table)
        response = dynamo_table.put_item(Item=data)
        return f"Success: Data inserted into table '{table}'. Response: {response}"
    except Exception as e:
        return f"Error inserting data into DynamoDB: {str(e)}"

class DynamoDBInsertAction(HyperbolicAction):
    """
    DynamoDB insert action for storing output data. This action can intelligently choose the DynamoDB
    table based on the provided data_type if the table name is not explicitly specified.
    """
    name: str = "dynamodb_insert"
    description: str = INSERT_PROMPT
    args_schema: type[BaseModel] = DynamoDBInsertInput
    func: Callable[..., str] = execute_dynamodb_insert
