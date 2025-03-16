import numpy as np
import pandas as pd

alternatives = [
    "1984", "Сто лет одиночества", "Преступление и наказание",
    "Мастер и Маргарита", "Хоббит", "Три товарища",
    "Шантарам", "Книжный вор", "Час презрения", "Игра престолов"
]

criteria_weights = np.array([5, 2, 3, 5, 4])
criteria_goal = np.array(["max", "min", "max", "min", "max"])

scores = np.array([
    [10, 5, 15, 10, 10],
    [10, 10, 5, 30, 5],
    [15, 10, 15, 5, 15],
    [15, 15, 15, 10, 10],
    [10, 5, 5, 50, 10],
    [10, 15, 15, 10, 10],
    [5, 15, 5, 50, 5],
    [15, 10, 15, 5, 15],
    [10, 5, 5, 50, 15],
    [15, 10, 15, 10, 15]
])

num_alternatives = len(alternatives)
preference_matrix = np.full((num_alternatives, num_alternatives), '-', dtype=object)


def compare_alternatives_1():
    results = ["Таким образом, имеем:"]

    for i in range(num_alternatives):
        for j in range(i + 1, num_alternatives):
            P_ij, N_ij, P_ji, N_ji = [0] * len(criteria_weights), [0] * len(criteria_weights), [0] * len(
                criteria_weights), [0] * len(criteria_weights)

            for k in range(len(criteria_weights)):
                if scores[i][k] != scores[j][k]:
                    if (criteria_goal[k] == "max" and scores[i][k] > scores[j][k]) or \
                            (criteria_goal[k] == "min" and scores[i][k] < scores[j][k]):
                        P_ij[k] = criteria_weights[k]
                        N_ji[k] = criteria_weights[k]
                    else:
                        P_ji[k] = criteria_weights[k]
                        N_ij[k] = criteria_weights[k]

            sum_P_ij, sum_N_ij = sum(P_ij), sum(N_ij)
            sum_P_ji, sum_N_ji = sum(P_ji), sum(N_ji)

            D_ij = sum_P_ij / sum_N_ij if sum_N_ij > 0 else float('inf')
            D_ji = sum_P_ji / sum_N_ji if sum_N_ji > 0 else float('inf')

            results.append(f"Рассмотрим альтернативы {i + 1} и {j + 1} (i={i + 1}, j={j + 1}):")
            results.append(f"P{i + 1}{j + 1} = {' + '.join(map(str, P_ij))} = {sum_P_ij};")
            results.append(f"N{i + 1}{j + 1} = {' + '.join(map(str, N_ij))} = {sum_N_ij};")
            results.append(
                f"D{i + 1}{j + 1} = P{i + 1}{j + 1} / N{i + 1}{j + 1} = {sum_P_ij}/{sum_N_ij} = {D_ij:.2f} {'> 1 — принимаем' if D_ij > 1 else '< 1 — отбрасываем'};")
            results.append(f"P{j + 1}{i + 1} = {' + '.join(map(str, P_ji))} = {sum_P_ji};")
            results.append(f"N{j + 1}{i + 1} = {' + '.join(map(str, N_ji))} = {sum_N_ji};")
            results.append(
                f"D{j + 1}{i + 1} = P{j + 1}{i + 1} / N{j + 1}{i + 1} = {sum_P_ji}/{sum_N_ji} = {D_ji:.2f} {'> 1 — принимаем' if D_ji > 1 else '< 1 — отбрасываем'};")
            results.append("")

    return "\n".join(results)


def compare_alternatives_2(c=None):
    for i in range(num_alternatives):
        for j in range(i + 1, num_alternatives):
            P_ij, N_ij, P_ji, N_ji = [0] * len(criteria_weights), [0] * len(criteria_weights), [0] * len(
                criteria_weights), [0] * len(criteria_weights)

            for k in range(len(criteria_weights)):
                if scores[i][k] != scores[j][k]:
                    if (criteria_goal[k] == "max" and scores[i][k] > scores[j][k]) or \
                            (criteria_goal[k] == "min" and scores[i][k] < scores[j][k]):
                        P_ij[k] = criteria_weights[k]
                        N_ji[k] = criteria_weights[k]
                    else:
                        P_ji[k] = criteria_weights[k]
                        N_ij[k] = criteria_weights[k]

            sum_P_ij, sum_N_ij = sum(P_ij), sum(N_ij)
            sum_P_ji, sum_N_ji = sum(P_ji), sum(N_ji)

            D_ij = sum_P_ij / sum_N_ij if sum_N_ij > 0 else float('inf')
            D_ji = sum_P_ji / sum_N_ji if sum_N_ji > 0 else float('inf')

            if c == None:
                if D_ij > 1:
                    preference_matrix[i, j] = round(D_ij, 2) if D_ij != float('inf') else '∞'
                if D_ji > 1:
                    preference_matrix[j, i] = round(D_ji, 2) if D_ji != float('inf') else '∞'
            else:
                if D_ij > 1 and D_ij > c:
                    preference_matrix[i, j] = round(D_ij, 2) if D_ij != float('inf') else '∞'
                elif 1 < D_ij < c:
                    preference_matrix[i, j] = '-'

                if D_ji > 1 and D_ji > c:
                    preference_matrix[j, i] = round(D_ji, 2) if D_ji != float('inf') else '∞'
                elif 1 < D_ji < c:
                    preference_matrix[i, j] = '-'


def print_preference_matrix(c=None):
    df = pd.DataFrame(preference_matrix, index=[i + 1 for i in range(num_alternatives)],
                      columns=[i + 1 for i in range(num_alternatives)])
    if c == None:
        print("\nСоставлена матрица предпочтений с внесенными и принятыми значениями D:\n")
    else:
        print(f"\nСоставлена матрица предпочтений с внесенными и принятыми значениями D, с порогом C = {c}:\n")
    print(df.to_string())


def find_best_alternative():
    # Считаем количество предпочтений для каждой альтернативы
    preference_counts = np.zeros(num_alternatives, dtype=int)

    for i in range(num_alternatives):
        for j in range(num_alternatives):
            if preference_matrix[i, j] != '-' and preference_matrix[i, j] != 0:
                preference_counts[i] += 1

    # Находим максимальное количество предпочтений
    max_preference = np.max(preference_counts)

    # Находим все альтернативы с максимальным количеством предпочтений
    best_alternatives = [alternatives[i] for i, count in enumerate(preference_counts) if count == max_preference]

    return best_alternatives, preference_counts


print(compare_alternatives_1())
compare_alternatives_2()
print_preference_matrix()
# Вывод лучших альтернатив
best_alternatives, preference_counts = find_best_alternative()
print("\nЛучшие альтернативы:", best_alternatives)
print("Количество предпочтений для каждой альтернативы:", preference_counts)
