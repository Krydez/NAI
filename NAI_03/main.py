"""
Silnik rekomendacji filmów wykorzystujący collaborative filtering.

Instrukcja przygotowania środowiska:
pip:
  python main.py
uv:
  uv run main.py

Autorzy: Hubert Jóźwiak, Kacper Olejnik
"""

from collections import defaultdict
from math import sqrt


def load_ratings(filename: str) -> dict:
    """
    Wczytuje oceny z pliku CSV.

    Args:
        filename: Ścieżka do pliku CSV z ocenami

    Returns:
        Słownik {użytkownik: {film: ocena}}
    """
    ratings = defaultdict(dict)

    with open(filename, "r", encoding="utf-8") as file:
        # Pomiń nagłówek
        next(file)

        for line in file:
            # Podziel linię na części
            parts = line.strip().split(",")

            # Użytkownik to pierwsza część, ocena to ostatnia, film to wszystko pomiędzy
            user = parts[0]
            rating = float(parts[-1])
            movie = ",".join(parts[1:-1])  # Połącz wszystkie środkowe części z powrotem

            ratings[user][movie] = rating

    return dict(ratings)


def cosine_similarity(user1_ratings: dict, user2_ratings: dict) -> float:
    """
    Oblicza podobieństwo kosinusowe między dwoma użytkownikami.

    Args:
        user1_ratings: Oceny pierwszego użytkownika
        user2_ratings: Oceny drugiego użytkownika

    Returns:
        Wartość podobieństwa (0-1, gdzie 1 oznacza identyczne gusta)
    """
    # Znajdź wspólne filmy
    common_movies = set(user1_ratings.keys()) & set(user2_ratings.keys())

    if not common_movies:
        return 0.0

    # Oblicz iloczyn skalarny i normy
    dot_product = sum(
        user1_ratings[movie] * user2_ratings[movie] for movie in common_movies
    )
    norm1 = sqrt(sum(user1_ratings[movie] ** 2 for movie in common_movies))
    norm2 = sqrt(sum(user2_ratings[movie] ** 2 for movie in common_movies))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def find_similar_users(target_user: str, all_ratings: dict, n: int = 5) -> list:
    """
    Znajduje najbardziej podobnych użytkowników do użytkownika docelowego.

    Args:
        target_user: Użytkownik, dla którego szukamy podobnych
        all_ratings: Wszystkie oceny użytkowników
        n: Liczba podobnych użytkowników do znalezienia

    Returns:
        Lista krotek (użytkownik, podobieństwo) posortowana malejąco
    """
    if target_user not in all_ratings:
        return []

    target_ratings = all_ratings[target_user]
    similarities = []

    for user, ratings in all_ratings.items():
        if user != target_user:
            similarity = cosine_similarity(target_ratings, ratings)
            if similarity > 0:
                similarities.append((user, similarity))

    # Sortuj malejąco po podobieństwie
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:n]


def get_recommendations(
    target_user: str, all_ratings: dict, n_recommendations: int = 5
) -> list:
    """
    Generuje rekomendacje filmów dla użytkownika.

    Args:
        target_user: Użytkownik, dla którego generujemy rekomendacje
        all_ratings: Wszystkie oceny użytkowników
        n_recommendations: Liczba rekomendacji do wygenerowania

    Returns:
        Lista krotek (film, przewidywana_ocena) posortowana malejąco
    """
    if target_user not in all_ratings:
        return []

    target_ratings = all_ratings[target_user]
    watched_movies = set(target_ratings.keys())

    # Znajdź podobnych użytkowników
    similar_users = find_similar_users(target_user, all_ratings, n=10)

    if not similar_users:
        return []

    # Oblicz przewidywane oceny dla nieobejrzanych filmów
    movie_scores = defaultdict(lambda: {"sum": 0.0, "weight_sum": 0.0})

    for similar_user, similarity in similar_users:
        for movie, rating in all_ratings[similar_user].items():
            if movie not in watched_movies:
                # Ważona suma ocen (wyższe oceny od bardziej podobnych użytkowników mają większą wagę)
                movie_scores[movie]["sum"] += rating * similarity
                movie_scores[movie]["weight_sum"] += similarity

    # Oblicz średnie ważone oceny
    recommendations = []
    for movie, scores in movie_scores.items():
        if scores["weight_sum"] > 0:
            predicted_rating = scores["sum"] / scores["weight_sum"]
            recommendations.append((movie, predicted_rating))

    # Sortuj malejąco po przewidywanej ocenie
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:n_recommendations]


def get_anti_recommendations(
    target_user: str, all_ratings: dict, n_recommendations: int = 5
) -> list:
    """
    Generuje antyrekomendacje (filmy, których użytkownik nie powinien oglądać).

    Args:
        target_user: Użytkownik, dla którego generujemy antyrekomendacje
        all_ratings: Wszystkie oceny użytkowników
        n_recommendations: Liczba antyrekomendacji do wygenerowania

    Returns:
        Lista krotek (film, przewidywana_ocena) posortowana rosnąco
    """
    if target_user not in all_ratings:
        return []

    target_ratings = all_ratings[target_user]
    watched_movies = set(target_ratings.keys())

    # Znajdź podobnych użytkowników
    similar_users = find_similar_users(target_user, all_ratings, n=10)

    if not similar_users:
        return []

    # Oblicz przewidywane oceny dla nieobejrzanych filmów
    movie_scores = defaultdict(lambda: {"sum": 0.0, "weight_sum": 0.0})

    for similar_user, similarity in similar_users:
        for movie, rating in all_ratings[similar_user].items():
            if movie not in watched_movies:
                movie_scores[movie]["sum"] += rating * similarity
                movie_scores[movie]["weight_sum"] += similarity

    # Oblicz średnie ważone oceny
    anti_recommendations = []
    for movie, scores in movie_scores.items():
        if scores["weight_sum"] > 0:
            predicted_rating = scores["sum"] / scores["weight_sum"]
            anti_recommendations.append((movie, predicted_rating))

    # Sortuj rosnąco po przewidywanej ocenie (najniższe oceny na górze)
    anti_recommendations.sort(key=lambda x: x[1])
    return anti_recommendations[:n_recommendations]


def main():
    # Wczytaj oceny z pliku
    ratings = load_ratings("ratings.csv")

    # Wyświetl dostępnych użytkowników
    print(f"\nDostępni użytkownicy ({len(ratings)}):")
    users_list = sorted(ratings.keys())
    for i, user in enumerate(users_list, 1):
        print(f"  {i}. {user}")

    # Interaktywny wybór użytkownika
    while True:
        try:
            choice = input(
                f"\nWybierz numer użytkownika (1-{len(users_list)}): "
            ).strip()
            if choice == "":
                print("Błąd: Należy wybrać numer użytkownika.")
                continue

            choice_num = int(choice)
            if 1 <= choice_num <= len(users_list) or choice == "":
                target_user = users_list[choice_num - 1]
                break
            else:
                print(f"Błąd: Wybierz numer z zakresu 1-{len(users_list)}")
        except ValueError:
            print("Błąd: Wprowadź liczbę lub naciśnij Enter")
        except KeyboardInterrupt:
            print("\n\nProgram przerwany przez użytkownika.")
            return

    # Wygeneruj rekomendacje
    recommendations = get_recommendations(target_user, ratings, n_recommendations=5)
    print(f"Top 5 filmów rekomendowanych dla {target_user}:")
    for i, (movie, score) in enumerate(recommendations, 1):
        print(f"  {i}. {movie}")
        print(f"     Przewidywana ocena: {score:.2f}/10")

    # Wygeneruj antyrekomendacje
    anti_recommendations = get_anti_recommendations(
        target_user, ratings, n_recommendations=5
    )
    print(f"\nTop 5 filmów, których {target_user} powinien unikać:")
    for i, (movie, score) in enumerate(anti_recommendations, 1):
        print(f"  {i}. {movie}")
        print(f"     Przewidywana ocena: {score:.2f}/10")


if __name__ == "__main__":
    main()
