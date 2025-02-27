# testing file for posting data to dynamodb

import os
import boto3
from decimal import Decimal
import json
from dotenv import load_dotenv
load_dotenv()
# Load environment variables from a specific .env file
# Adjust the path if your .env file is located in a different directory

def convert_floats_to_decimals(obj):
    """
    Recursively convert all float values in the object to Decimal.
    """
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: convert_floats_to_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_decimals(item) for item in obj]
    else:
        return obj

def insert_to_dynamodb(data, table_name, region_name="us-east-2"):
    """
    Insert data into a DynamoDB table after converting floats to Decimal types.
    Credentials are loaded from environment variables.
    
    Args:
        data (dict): The data to insert.
        table_name (str): Name of the DynamoDB table.
        region_name (str): AWS region where the table is located.
    
    Returns:
        str: Success message or error message.
    """
    # Load AWS credentials from environment
    ACCESS_KEY = os.getenv("ACCESS_KEY")
    SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")
    if not ACCESS_KEY or not SECRET_ACCESS_KEY:
        return "Error: ACCESS_KEY and/or SECRET_ACCESS_KEY not set in the environment."

    # Convert all float values in the data to Decimal
    data = convert_floats_to_decimals(data)
    
    try:
        # Initialize the DynamoDB resource
        dynamodb = boto3.resource(
            "dynamodb",
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_ACCESS_KEY,
            region_name=region_name
        )
        table = dynamodb.Table(table_name)
        response = table.put_item(Item=data)
        return f"Success: Data inserted into table '{table_name}'. Response: {response}"
    except Exception as e:
        return f"Error inserting data into DynamoDB: {str(e)}"

if __name__ == "__main__":
    # Dummy data for testing the insert operation
    dummy_data = {
        "gpu_health": {
            "gpu_info": {
                "uuid": "GPU-dummy",
                "name": "Dummy GPU",
                "driver_version": "1.0"
            },
            "metrics": {
                "temperature": {
                    "value": 30,
                    "unit": "C",
                    "status": "normal"
                },
                "fan_speed": {
                    "value": "[N/A]",
                    "unit": "%"
                },
                "memory": {
                    "used": 1,
                    "total": 1000,
                    "unit": "MB",
                    "utilization": 0,
                    "status": "normal"
                },
                "gpu_utilization": {
                    "value": 0,
                    "unit": "%",
                    "status": "normal"
                },
                "power": {
                    "draw": 50.0,
                    "limit": 100.0,
                    "unit": "W",
                    "utilization": 50.0,
                    "status": "normal"
                }
            },
            "concerns": [],
            "status_messages": [
                "All metrics normal"
            ],
            "overall_status": "healthy",
            "timestamp": "2025-02-27T14:00:00.000000",
            "flagged": False
        },
        "status": "healthy"
    }
    
    # Specify your DynamoDB table name (must exist in your AWS account in the specified region)
    table_name = "GPUHealthChecks"
    
    # Optionally adjust the region if your table is in a different region
    region_name = "us-east-2"  # Change if needed
    
    # Insert dummy data into DynamoDB and print the result
    result = insert_to_dynamodb(dummy_data, table_name, region_name)
    print(result)
