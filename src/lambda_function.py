from utils import PokemonModel
from tests.test_handler import sample_data
import json
import sys


def lambda_handler(event, context):
    """Main handler interface for model, accepts payload and passes to functions"""

    search_path = sys.path
    print(search_path)
    print("checkpoint 1 - Path initialised")

    pokemon_model = PokemonModel()
    print("checkpoint 2 - Model successfully instantiated")

    body = json.loads(event.get('body', '{}'))

    if isinstance(body, dict):
        body = json.dumps(body)
    else:
        body = json.loads(body)

    body = json.loads(body)
    action = body.get('action', '').strip().lower()
    print(f"Action received: {action}")

    if action == 'recommend_by_type':
        pokemon_type = body.get('pokemon_type', '').strip().lower()
        recommendation = pokemon_model.recommend_pokemon_by_type(pokemon_type)
    elif action == 'recommend_by_stats':
        category = body.get('category', '').strip().lower()
        recommendation = pokemon_model.recommend_pokemon_by_stats_category(category)
    else:
        recommendation = "Invalid action. Use 'recommend_by_type' or 'recommend_by_stats'."

    # Prepare the response
    response = {
        'statusCode': 200,
        'body': json.dumps({'recommendation': recommendation})
    }
    print(f"Response: {response}")
    return response


if __name__ == '__main__':

    lambda_handler(sample_data["sample_event"], sample_data["sample_context"])