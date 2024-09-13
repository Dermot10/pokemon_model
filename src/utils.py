import joblib
import pandas as pd
import random


class PokemonModel:
    def __init__(self):
        self.model = joblib.load('../models/pokemon_model.pkl')
        self.preprocessor = joblib.load('../models/pokemon_preprocessor.pkl')
        self.dataset_path = '../datasets/pokemon_complete_dataset.csv'
        self.dataframe = pd.read_csv(self.dataset_path)
        self.df_stats = self.dataframe['total_stats'] = self.dataframe[['stats_hp', 'stats_attack', 'stats_defense',
                                                                        'stats_special-attack', 'stats_special-defense',
                                                                        'stats_speed']].sum(axis=1)

    def categorize_stats(self, total):
        """Helper function to categorize stats"""
        if total >= 580:
            return 'Elite'
        elif total >= 450:
            return 'Strong'
        elif total >= 340:
            return 'Medium'
        else:
            return 'Weak'

    def recommend_pokemon_by_stats_category(self, category):
        """Map category to the corresponding stat range"""
        if category == 'Elite'.lower():
            stat_range = (580, float('inf'))
        elif category == 'Strong'.lower():
            stat_range = (450, 579)
        elif category == 'Medium'.lower():
            stat_range = (340, 449)
        elif category == 'Weak'.lower():
            stat_range = (0, 339)
        else:
            return "Invalid category."

        # Filter the dataset based on the total stats range
        filtered_df = self.dataframe[(self.dataframe['total_stats'] >= stat_range[0]) & (self.dataframe['total_stats'] <= stat_range[1])]

        if filtered_df.empty:
            return "No Pokémon found in the specified category."

        # Randomly select a Pokémon from the filtered DataFrame
        return random.choice(filtered_df['name'].tolist())

    def recommend_pokemon_by_type(self, pokemon_type):
        """Filter the dataset based on the requested type in either type_1 or type_2"""

        filtered_df = self.dataframe[
            (self.dataframe['type_1'].str.lower() == pokemon_type.lower()) | (self.dataframe['type_2'].str.lower() == pokemon_type.lower())]

        if filtered_df.empty:
            return f"No Pokémon found for type '{pokemon_type}'."

        # Randomly select a Pokémon from the filtered DataFrame
        return random.choice(filtered_df['name'].tolist())

