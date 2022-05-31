from neural_networks import *
from selection import *
from crossover import *
from mutation import *
from classes import *
from game_configurations import *

# n_input -> no. of input units
# n_hidden1 -> no. of units in hidden layer 1
# n_hidden1 -> no. of units in hidden layer 2
# n_output -> no. of output units

# The population will have pop_size chromosome where each chromosome has n_weights genes.
n_weights = n_input * n_hidden1 + n_hidden1 * n_hidden2 + n_hidden2 * n_output


#############################################################################################
def get_fitness(self):
    # Initialize the values of the fitness function
    max_score, death, total_steps = 0, 0, 0
    test_games = 1
    steps_per_game = 2000

    # Variables to penalize and award the snakes
    steps_without_eating, penaltyHunger, penaltyLoop, changed_direction, awardFront = 0, 0, 0, 0, 0
    still_front, still_left, still_right = 0, 0, 0

    for _ in range(test_games):
        steps_without_eating += 1
        total_steps += 1

        snake_start, snake_position, apple_position, score = starting_positions()

        for _ in range(steps_per_game):
            current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked = blocked_directions(
                snake_position)
            angle, snake_direction_vector, apple_direction_vector_normalized, snake_direction_vector_normalized = angle_with_apple(
                snake_position, apple_position)
            predictions = []

            fa = isFoodAhead(snake_position, apple_position)
            if fa[0]:
                predicted_direction = fa[1]

            else:
                # Returns the best move --> output of neural network
                predicted_direction = np.argmax(np.array(forward_propagation(np.array(
                    [is_left_blocked, is_front_blocked, is_right_blocked, apple_direction_vector_normalized[0],
                     snake_direction_vector_normalized[0], apple_direction_vector_normalized[1],
                     snake_direction_vector_normalized[1]]).reshape(-1, 7), self.representation))) - 1

            if predicted_direction == 0:  # front
                still_front += 1
                still_left, still_right = 0, 0

                if still_front >= 3:
                    awardFront += 1

            elif predicted_direction == -1:  # left
                still_left += 1
                still_right, still_front = 0, 0
                changed_direction += 1

            elif predicted_direction == 1:  # right
                still_right += 1
                still_left, still_front = 0, 0
                changed_direction += 1

            if (still_left > 40) or (still_right > 40):  # 10 times in a loop
                penaltyLoop += 1
                still_left, still_right = 0, 0

            new_direction = np.array(snake_position[0]) - np.array(snake_position[1])
            if predicted_direction == -1:
                new_direction = np.array([new_direction[1], -new_direction[0]])
            if predicted_direction == 1:
                new_direction = np.array([-new_direction[1], new_direction[0]])

            button_direction = generate_button_direction(new_direction)

            next_step = snake_position[0] + current_direction_vector
            if collision_with_boundaries(snake_position[0]) == 1 or collision_with_self(next_step.tolist(),
                                                                                        snake_position) == 1:
                death += 1
                break

            snake_position, apple_position, score = play_game(snake_start, snake_position, apple_position,
                                                              button_direction, score, display, clock)

            if steps_without_eating > 200:
                penaltyHunger += 1
                steps_without_eating = 0

            if score > max_score:
                max_score = score
                steps_without_eating = 0

    fitness = max_score * 4000 - death * 100 - penaltyLoop * 5 - penaltyHunger * 100 + awardFront * 2
    return round(fitness, 2), max_score


#############################################################################################
# Monkey Patching !!!
Snake.get_fitness = get_fitness

# Objective is to maximize our fitness function
print(f' --- Beginning of Generation 0 ---')
pop = Population(size=100, optim="max", sol_size=n_weights, replacement=True)

pop.evolve(gens=50,
           select=tournament_new,
           crossover=crossover_mix,
           mutate=swap_mutation,
           co_p=0.9,
           mu_p=0.1,
           elitism=True,
           n_elit=5)

