import json
# Sample event data for testing, add to main file to test

# sample_event = {
#     "body": json.dumps({
#         "action": "recommend_by_type",
#         "pokemon_type": "water"
#     })
# }

# sample_event = {
#     "body": json.dumps({
#         "action": "recommend_by_type",
#         "pokemon_type": "dragon"
#     })
# }

# sample_event = {
#     "body": json.dumps({
#         "action": "recommend_by_type",
#         "pokemon_type": "electric"
#     })
# }

# sample_event = {
#     "body": json.dumps({
#         "action": "recommend_by_stats,
#         "category": "elite"
#     })
# }

sample_event = {
    "body": json.dumps({
        "action": "recommend_by_stats",
        "category": "strong"
    })
}

# sample_event = {
#     "body": json.dumps({
#         "action": "recommend_by_stats_category",
#         "category": "medium"
#     })
# }

# Context is not used in this example, but required for the Lambda function signature
sample_context = {}

# Test the function
sample_data = {"sample_event": sample_event, "sample_context": sample_context}


